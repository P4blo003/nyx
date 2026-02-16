# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from prometheus_client import Counter, Histogram, Gauge


# ==============================
# METRICS
# ==============================

REQUEST_COUNTER = Counter(
    "ai_service_requests_total",
    "Total requests",
    ["method", "status"]
)
REQUEST_LATENCY = Histogram(
    "ai_service_request_duration_seconds",
    "Request latency",
    ["method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
ACTIVE_REQUESTS = Gauge(
    "ai_service_active_requests",
    "Currently processing requests"
)
BATCH_SIZE = Histogram(
    "ai_service_batch_size",
    "Input batch sizes",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)
TRITON_CLIENT_ERRORS = Counter(
    "ai_service_triton_errors_total",
    "Triton client errors",
    ["endpoint", "error_type"]
)
