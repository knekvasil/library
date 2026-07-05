# Docker → k3s Migration Guide

This guide covers the migration of a server from a Docker Compose setup to a single-node k3s cluster.

## Background

The original setup ran on Docker Compose with four services:

- **card-game** (Node.js web game, port 2567) — containerized
- **stats-app** (Python data app, port 8008) — containerized
- **postgres** (postgres:18, port 5432) — containerized, shared via a Docker network
- **cloudflared** — two tunnels (card-game as systemd service, stats-app as user process)

CI/CD was a cron job polling Docker Hub every 5 minutes for new card-game images. No auto-deploy for stats-app. No monitoring. Docker build cache was consuming 27 GB.

## Migration Steps

### Phase 0: Preparation

1. **Safety dump of postgres:**
```bash
docker exec postgres-master pg_dumpall -U postgres > /home/user/pg-pre-migration.sql
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

1. **Create postgres StatefulSet** with 10 GB PVC, tuned config for server RAM
2. **Copy dump and restore:**
```bash
kubectl cp /home/user/pg-pre-migration.sql postgres-0:/tmp/ -n database
kubectl exec -i postgres-0 -n database -- psql -U postgres -f /tmp/pg-dump.sql
```
3. **Verify data:**
```bash
kubectl exec postgres-0 -n database -- psql -U postgres -d stats-app -c "SELECT count(*) FROM users;"
```
4. **Deploy backup CronJob** — weekly pg_dump to host-mounted volume

### Phase 2: Apps to k3s

For each app, create: Namespace, ConfigMap, Secret, Deployment, Service, Ingress.

1. **stats-app** (pilot — stateless, reads-only):
```bash
kubectl apply -f stats-app/
```

2. **card-game** (includes periodic CronJob):
```bash
kubectl apply -f card-game/
```

### Phase 3: Cloudflare Tunnel

1. **Create credential Secret** from existing tunnel JSON:
```bash
kubectl create secret generic tunnel-credentials --from-file=credentials.json=...
```

2. **Deploy cloudflared** with multi-hostname ingress config

3. **Stop old tunnels:**
```bash
systemctl stop cloudflared
kill $(pgrep -f "cloudflared tunnel.*stats-app")
```

### Phase 4: Monitoring

1. **Install kube-prometheus-stack** via Helm:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

2. **Set Grafana admin password** via a Kubernetes Secret

3. **Add disk alert** — PrometheusRule for >80% usage

4. **Configure alert notifications** — Alertmanager with email/Slack/etc.

### Phase 5: CI/CD

1. **Create GitHub deployments workflow** — build → registry push → SSH → `kubectl set image`

2. **Add GitHub Secrets:** registry PAT (with `write:packages`, `workflow`, `repo` scopes), SSH deploy key

3. **Replace old cron jobs** with k8s CronJobs

### Phase 6: Cleanup

1. **Stop Docker containers:**
```bash
docker rm -f card-game stats-app postgres-master
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
For design rationale and lessons learned, see [Single-Node k3s Architecture](servr-architecture.md).
