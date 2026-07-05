# Docker → k3s Migration Guide

This guide covers the migration of `triton` from a Docker Compose setup to a single-node k3s cluster.

## Background

The original setup ran on Docker Compose with four services:

- **gin13** (Node.js card game, port 2567) — containerized
- **elfant** (Python fantasy football app, port 8008) — containerized
- **postgres-master** (postgres:18, port 5432) — containerized, shared via `db-net`
- **cloudflared** — two tunnels (gin13 as systemd service, elfant as user process)

CI/CD was a cron job polling Docker Hub every 5 minutes for new gin13 images. No auto-deploy for elfant. No monitoring. Docker build cache was consuming 27 GB.

## Migration Steps

### Phase 0: Preparation

1. **Safety dump of postgres:**
```bash
docker exec postgres-master pg_dumpall -U postgres > /home/kaj/pg-pre-migration.sql
```

2. **Install k3s** (with Traefik, write-kubeconfig-mode 644):
```bash
curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
```

3. **Install k9s:**
```bash
curl -LO https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_amd64.tar.gz
tar -xzf k9s_Linux_amd64.tar.gz -C /usr/local/bin k9s
```

### Phase 1: Postgres to k3s

1. **Create postgres StatefulSet** with 10 GB PVC, tuned config for 8 GB RAM
2. **Copy dump and restore:**
```bash
kubectl cp /home/kj/pg-pre-migration.sql postgres-0:/tmp/ -n database
kubectl exec -i postgres-0 -n database -- psql -U postgres -f /tmp/pg-dump.sql
```
3. **Verify data:**
```bash
kubectl exec postgres-0 -n database -- psql -U postgres -d elfant -c "SELECT count(*) FROM users;"
```
4. **Deploy backup CronJob** — weekly pg_dump to `/var/lib/postgres-backups/`

### Phase 2: Apps to k3s

For each app, create: Namespace, ConfigMap, Secret, Deployment, Service, Ingress.

1. **elfant** (pilot — stateless, reads-only):
```bash
kubectl apply -f elfant/
```

2. **gin13** (includes bot-league CronJob):
```bash
kubectl apply -f gin13/
```

### Phase 3: Cloudflare Tunnel

1. **Create credential Secret** from existing tunnel JSON:
```bash
kubectl create secret generic tunnel-triton --from-file=credentials.json=...
```

2. **Deploy cloudflared** with multi-hostname ingress config

3. **Rename tunnel** (gin13 → triton-tunnel):
```bash
curl -X PATCH https://api.cloudflare.com/client/v4/accounts/.../cfd_tunnel/$TUNNEL_ID \
  -d '{"name": "triton-tunnel"}'
```

4. **Stop old tunnels:**
```bash
systemctl stop cloudflared
kill $(pgrep -f "cloudflared tunnel.*elfant")
```

### Phase 4: Monitoring

1. **Install kube-prometheus-stack** via Helm:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.adminPassword=admin
```

2. **Set Grafana credentials** to `kaj` / `nek`

3. **Add disk alert** — PrometheusRule for >80% usage

4. **Configure email notifications** — Alertmanager with Gmail SMTP

### Phase 5: CI/CD

1. **Create GitHub deployments workflow** — build → GHCR push → SSH → `kubectl set image`

2. **Add GitHub Secrets:** `GHCR_PAT`, `DEPLOY_SSH_KEY`

3. **Replace old cron jobs** with k8s CronJobs

### Phase 6: Cleanup

1. **Stop Docker containers:**
```bash
docker rm -f gin13 elfant postgres-master
```

2. **Clear crontab** — all entries replaced

3. **Prune Docker artifacts:**
```bash
docker system prune -a -f
```

4. **Disable cloudflared systemd service:**
```bash
systemctl disable cloudflared
```

## Key Decisions

| Decision | Rationale |
|---|---|
| k3s over k8s | Single node, lightweight, but full k8s API |
| Traefik kept (not disabled) | Cloudflare tunnel bypasses it, but useful for future services |
| hostNetwork on pods | Postgres accessible directly on 127.0.0.1:5432 from pods |
| GHCR over Docker Hub | Source code on GitHub, fewer registries to manage |
| Local postgres backups | Practice over perfection — same machine, but better than nothing |

## Lessons Learned

- **Postgres 18** changed data directory naming — mount at `/var/lib/postgresql` not `/var/lib/postgresql/data`
- **Cloudflared metrics** binds to localhost by default — pass `--metrics 0.0.0.0:2000` for liveness probes
- **Renaming a tunnel** via Cloudflare API doesn't change the tunnel ID or credentials — only the name
- **GitHub PAT needs `workflow` scope** to push `.github/workflows/` files
- `docker system prune -a` reclaimed 48 GB of unused images + build cache
