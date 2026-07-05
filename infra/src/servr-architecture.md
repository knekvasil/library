# Single-Node k3s Architecture

## Overview

A single-node **k3s** cluster running on a Linux server. All workloads are Kubernetes pods. All inbound traffic routes through a single Cloudflare Tunnel — no ports open on the public internet.

```
                          Cloudflare Edge
                               |
                  CNAME → cloudflare tunnel
                               |
                      cloudflared pod
                     (cloudflared ns)
                     ────┬────┬────
                         │    │    │
                    ┌────┘    │    └────┐
                    ▼         ▼         ▼
                 app-1:PORT  app-2:PORT  grafana:PORT
               (app-1 ns)  (app-2 ns)  (monitoring ns)
                    │         │
                    └────┬────┘
                         ▼
                   postgres:5432
                 (database ns, StatefulSet)
```

All three services route through a single Cloudflare Tunnel (running as a pod in the `cloudflared` namespace). The tunnel handles hostname-based routing to the correct internal service.

## Routing

| Domain | Target Service | Namespace |
|---|---|---|
| `app-1.yourdomain.com` | `app-1` Deployment | `app-1` |
| `app-2.yourdomain.com` | `app-2` Deployment | `app-2` |
| `monitor.yourdomain.com` | `grafana` Service | `monitoring` |

Each domain is configured as a public hostname in the Cloudflare Tunnel dashboard, pointing to `service:port` of the corresponding Kubernetes Service.

## Namespaces

| Namespace | Resources |
|---|---|
| `database` | PostgreSQL StatefulSet (1 replica, persistent PVC), backup CronJob |
| `app-1` | App Deployment, Service, Ingress, periodic CronJob |
| `app-2` | App Deployment, Service, Ingress, periodic CronJob |
| `cloudflared` | `cloudflared` Deployment, tunnel ConfigMap + credential Secret |
| `monitoring` | Prometheus + Grafana + Alertmanager + node-exporter + kube-state-metrics |
| `kube-system` | CoreDNS, Traefik, metrics-server, local-path-provisioner |

## PostgreSQL

### Deployment
- PostgreSQL deployed as a **StatefulSet** in the `database` namespace
- Persistent volume via local-path provisioner (or your preferred storage class)
- Credentials stored in a Kubernetes Secret
- Databases are created per application
- In-cluster DNS: `postgres.database:5432`

### Backup Strategy
- CronJob runs weekly `pg_dump` to a host-mounted volume
- Backups are gzipped SQL
- Retention: last 4 backups
- Restore: `gunzip -c backup.sql.gz | kubectl exec -i postgres-0 -n database -- psql -U postgres`

## CI/CD

### Deployment Pipeline
Push to `master` triggers a GitHub Actions workflow:

1. Build Docker image (tagged `:latest` and `:$SHA`)
2. Push to container registry (e.g., `ghcr.io/<user>/<repo>`)
3. SSH into server → `kubectl set image deployment/<app> <container>=<new-image>`
4. `kubectl rollout status` — fails if rollout doesn't complete within a timeout

**Required GitHub Secrets:** registry token (with `write:packages`, `workflow`, `repo` scopes), SSH deploy key.

### Path Triggers
Workflows can be scoped to only run when relevant files change:
- **app-1:** `server/**`, `client/**`, `shared/**`, `Dockerfile`
- **app-2:** `app/**`, `Dockerfile`, `frontend/**`

## CronJobs

| Name | Schedule | Purpose |
|---|---|---|
| `app-1-task` | Daily 6 AM | Runs a script inside the app container |
| `app-2-sync` | Weekly Wed 12 PM | Syncs data via a CLI command |
| `postgres-backup` | Weekly Sun 5 AM | `pg_dump` → gzip |

No system-level crontab entries are used — all scheduled tasks run as Kubernetes CronJobs.

## Monitoring

**Grafana** is deployed via the kube-prometheus-stack Helm chart:

- Prometheus scrapes node-exporter and kube-state-metrics
- Default dashboards: Node Exporter / Nodes, Kubernetes / Pods
- **Disk alert:** PrometheusRule fires when disk usage exceeds 80% for 5+ minutes
- **Alertmanager** handles routing (email, Slack, etc.)

**k9s** TUI: run `k9s` on the server for live pod/resource management.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| k3s over full k8s | Single node, lightweight, same k8s API |
| Cloudflare Tunnel over open ports | No public IP exposure, DDoS protection, free |
| Traefik kept | Useful if internal ingress routing is needed later |
| hostNetwork on pods | Direct localhost access to databases from other pods |
| GHCR (or your registry) over Docker Hub | Tight GitHub integration, fewer registries |
| Local backups | Single-node limitation — offsite backup recommended for production |

## Lessons Learned

- **Postgres version changes** may rename the data directory — verify the expected mount path before setting up PVCs
- **Cloudflared metrics** bind to localhost by default — pass `--metrics 0.0.0.0:PORT` for liveness probes
- **Renaming a tunnel** via Cloudflare API doesn't change the tunnel ID or credentials — only the display name
- **GitHub PAT needs `workflow` scope** to push `.github/workflows/` files
- **Docker system prune** can reclaim significant disk space after migration away from Docker Compose
