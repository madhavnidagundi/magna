#!/usr/bin/env python3
"""
Magna Metrics Simulator

Exposes the same Prometheus metrics that the real Magna middleware produces,
with realistic-looking values. This lets the maintainer see the full Grafana
dashboard with live data without needing to build/run the Rust middleware.

Metrics exposed on :9090/metrics (same as the real middleware):
  - inference_requests_total  (counter, per-device)
  - inference_latency_milliseconds (histogram, per-device)
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import random
import threading
import time

# ── Metric State ────────────────────────────────────────────────────────────

devices = ["cpu"]
request_counts = {d: 0 for d in devices}
latency_sums = {d: 0.0 for d in devices}
latency_counts = {d: 0 for d in devices}

# Histogram buckets matching OpenTelemetry SDK defaults
BUCKETS = [0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
bucket_counts = {d: {b: 0 for b in BUCKETS} for d in devices}
bucket_counts_inf = {d: 0 for d in devices}

lock = threading.Lock()


def simulate_request(device: str) -> None:
    """Simulate a single inference request with realistic latency."""
    # Realistic latency: mostly 15-45ms with occasional spikes
    if random.random() < 0.05:
        latency = random.uniform(80, 200)  # occasional spike
    else:
        latency = random.gauss(30, 8)  # normal inference
        latency = max(5, latency)

    with lock:
        request_counts[device] += 1
        latency_sums[device] += latency
        latency_counts[device] += 1
        for b in BUCKETS:
            if latency <= b:
                bucket_counts[device][b] += 1
        bucket_counts_inf[device] += 1


def background_traffic() -> None:
    """Generate simulated traffic in the background."""
    while True:
        for device in devices:
            simulate_request(device)
        # ~15-25 requests per second
        time.sleep(random.uniform(0.04, 0.07))


# ── Prometheus Text Format ──────────────────────────────────────────────────

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        lines = []

        # inference_requests_total (counter)
        lines.append("# HELP inference_requests_total Total number of inference requests")
        lines.append("# TYPE inference_requests_total counter")
        with lock:
            for device in devices:
                lines.append(
                    f'inference_requests_total{{device="{device}",'
                    f'otel_scope_name="magna-middleware",'
                    f'service_name="magna-server"}} {request_counts[device]}'
                )

        # inference_latency_milliseconds (histogram)
        lines.append("")
        lines.append("# HELP inference_latency_milliseconds Latency of inference requests in milliseconds")
        lines.append("# TYPE inference_latency_milliseconds histogram")
        with lock:
            for device in devices:
                attrs = f'device="{device}",otel_scope_name="magna-middleware",service_name="magna-server"'
                for b in BUCKETS:
                    lines.append(
                        f"inference_latency_milliseconds_bucket{{{attrs},le=\"{b}\"}} {bucket_counts[device][b]}"
                    )
                lines.append(
                    f'inference_latency_milliseconds_bucket{{{attrs},le="+Inf"}} {bucket_counts_inf[device]}'
                )
                lines.append(
                    f"inference_latency_milliseconds_sum{{{attrs}}} {latency_sums[device]:.2f}"
                )
                lines.append(
                    f"inference_latency_milliseconds_count{{{attrs}}} {latency_counts[device]}"
                )

        body = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args) -> None:
        """Suppress per-request logs to keep output clean."""
        pass


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start background traffic generator
    t = threading.Thread(target=background_traffic, daemon=True)
    t.start()

    port = 9090
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"Magna metrics simulator running on http://0.0.0.0:{port}/metrics")
    print("Generating ~15-25 simulated inference requests/sec...")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
