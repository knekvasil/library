# SERVR Architecture

> Hostname: `triton` · Domain: `kajnekvasil.com` · IP: `192.168.1.206`
> CPU: Intel i5-7360U (2C/4T) · RAM: 7.6 GiB · Disk: 98 GB LVM

## Overview

Single-node **k3s** cluster. No Docker. All workloads are Kubernetes pods. All inbound traffic goes through a single Cloudflare tunnel.

```
                         Cloudflare Edge
                              │
                 CNAME → triton-tunnel
                              │
                     cloudflared pod
                    (cloudflared ns)
                    ────┬────┬────
                        │    │    │
                   ┌────┘    │    └────┐
                   ▼         ▼         ▼
              gin13:2567  elfant:8008  grafana:80
            (gin13 ns)  (elfant ns)  (monitoring ns)
                   │         │
                   └────┬────┘
                        ▼
                  postgres:5432
                (database ns, StatefulSet)
```

## Routing

| Domain | Service | Namespace | Port |
|---|---|---|---|
| `gin13.kajnekvasil.com` | `gin13` | `gin13` | 2567 |
| `elfant.kajnekvasil.com` | `elfant` | `elfant` | 8008 |
| `monitor.kajnekvasil.com` | `prometheus-grafana` | `monitoring` | 80 |

All three route through tunnel `b6a7eea2-...` (`triton-tunnel`). The tunnel pod runs in `cloudflared` namespace.

## Namespaces

| Namespace | Resources |
|---|---|
| `database` | `postgres` StatefulSet (1 replica, 10 GB PVC), `postgres-backup` CronJob |
| `elfant` | `elfant` Deployment, Service, Ingress, `sync-stats` CronJob |
| `gin13` | `gin13` Deployment, Service, Ingress, `bot-league` CronJob |
| `cloudflared` | `cloudflared` Deployment, tunnel ConfigMap + Secret |
| `monitoring` | Prometheus + Grafana + Alertmanager + node-exporter + kube-state-metrics |
| `kube-system` | CoreDNS, Traefik, metrics-server, local-path-provisioner |

## Postgres

- **Image:** `postgres:18` in `database` namespace
- **Storage:** 10 GB PVC via `local-path` (host: `/var/lib/rancher/k3s/storage/`)
- **Credentials:** k8s Secret `postgres-secret`
- **Databases:** `gin13`, `elfant`
- **Service DNS:** `postgres.database:5432`

### Backup
- CronJob `postgres-backup` runs Sunday 5 AM
- Saves to `/var/lib/postgres-backups/` on host
- Keeps last 4 backups, gzipped SQL
- Restore: `gunzip -c backup.sql.gz | kubectl exec -i postgres-0 -n database -- psql -U postgres`

## CI/CD

Push to `master` on either repo triggers:
1. Build Docker image (tagged `:latest` and `:$SHA`)
2. Push to `ghcr.io/knekvasil/<repo>`
3. SSH into server → `kubectl set image deployment/<app>`
4. `kubectl rollout status` — fails if rollout doesn't complete in 120s

**GitHub Secrets required:** `GHCR_PAT` (with `write:packages`, `workflow`, `repo`), `DEPLOY_SSH_KEY`

### Path triggers
- **gin13:** `server/**`, `client/**`, `shared/**`, `package.json`, `Dockerfile`
- **elfant:** `elfant/**`, `pyproject.toml`, `Dockerfile`, `frontend/**`

## CronJobs

| Name | Schedule | What |
|---|---|---|
| `bot-league` (`gin13`) | Daily 6 AM | `npx tsx bot-league.js` |
| `sync-stats` (`elfant`) | Wed 12 PM | `python3 -m elfant.cli sync-stats` |
| `postgres-backup` (`database`) | Sun 5 AM | `pg_dump` → gzip |

No system-level crontab entries remain.

## Monitoring

**Grafana** at `monitor.kajnekvasil.com` (login: `kaj` / `nek`).
- Home dashboard: **Node Exporter / Nodes**
- Prometheus + Alertmanager + kube-state-metrics
- Disk alert: fires email to `kajnekvasil@gmail.com` when >80% for 5+ minutes

**k9s** TUI: run `k9s` on the server for live pod/resource view.

## Key File Locations

| Path | What |
|---|---|
| `/home/kaj/k3s-manifests/` | All Kubernetes manifests |
| `/home/kaj/SERVR.md` | Concise reference (this content) |
| `/home/kaj/.ssh/deploy-key` | GitHub Actions SSH key |
| `/var/lib/postgres-backups/` | Postgres backup destination |
