# Magna Monitoring Stack

Pre-configured Prometheus + Grafana monitoring stack for the Magna inference middleware.  
**Runs fully self-contained** — includes a metrics simulator so you see live dashboard data immediately.

## Quick Start

```bash
cd grafana/
docker compose up -d --build
```

Open [http://localhost:3000](http://localhost:3000) → login `admin` / `admin` → the **Magna Inference Dashboard** is already loaded with live data.

That's it. No additional setup required.

## What's Running

| Service             | Port | Description                                               |
|---------------------|------|-----------------------------------------------------------|
| **metrics-simulator** | 9090 | Generates realistic inference metrics (same format as real middleware) |
| **Prometheus**      | 9091 | Scrapes metrics from the simulator                        |
| **Grafana**         | 3000 | Pre-loaded with the Magna Inference Dashboard             |

### Pre-provisioned Dashboard

The dashboard includes three panels, all showing live data from the simulator:

- **Inference Request Rate** (req/s) — per device
- **Average Inference Latency** (ms) — per device
- **Total Requests** — cumulative count per device

## Switching to the Real Middleware

When you're ready to monitor the actual Magna middleware instead of the simulator:

1. Edit [`prometheus/prometheus.yml`](prometheus/prometheus.yml) and change the scrape target:

```yaml
  - job_name: 'magna-middleware'
    static_configs:
      - targets: ['host.docker.internal:9090']   # real middleware on host
```

2. Optionally remove the simulator service from [`docker-compose.yml`](docker-compose.yml) or just stop it:

```bash
docker compose stop metrics-simulator
docker compose restart prometheus
```

## Configuration

All configurable values are in [`.env`](.env):

| Variable                | Default | Description                |
|-------------------------|---------|----------------------------|
| `GRAFANA_ADMIN_USER`    | `admin` | Grafana admin username     |
| `GRAFANA_ADMIN_PASSWORD`| `admin` | Grafana admin password     |
| `GRAFANA_PORT`          | `3000`  | Host port for Grafana      |
| `PROMETHEUS_PORT`       | `9091`  | Host port for Prometheus   |

> **⚠️ Change the default credentials for any non-local deployment.**

## Directory Structure

```
grafana/
├── .env                             # Configurable variables (ports, credentials)
├── docker-compose.yml               # Orchestrates all services
├── metrics-simulator/
│   ├── Dockerfile                   # Python-based metrics simulator
│   └── simulator.py                 # Generates realistic Prometheus metrics
├── grafana/
│   ├── Dockerfile                   # Grafana image with configs baked in
│   ├── dashboards/
│   │   └── magna.json               # Magna Inference Dashboard definition
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml        # Dashboard provider config
│       └── datasources/
│           └── datasource.yml       # Prometheus datasource config
└── prometheus/
    ├── Dockerfile                   # Prometheus image
    └── prometheus.yml               # Scrape configuration (volume-mounted)
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+ / `docker compose` plugin)

## Common Commands

```bash
# Start everything
docker compose up -d --build

# View logs
docker compose logs -f

# View logs for a specific service
docker compose logs -f grafana

# Stop the stack
docker compose down

# Stop and remove all data (volumes)
docker compose down -v

# Restart after config changes
docker compose restart prometheus
```

## Troubleshooting

### Grafana shows "No Data"

1. Check that all containers are running: `docker compose ps`
2. Check Prometheus targets at [http://localhost:9091/targets](http://localhost:9091/targets) — the `magna-middleware` target should be UP
3. View simulator logs: `docker compose logs metrics-simulator`

### Port conflicts

Change ports in `.env`:

```bash
GRAFANA_PORT=3001
PROMETHEUS_PORT=9092
```
