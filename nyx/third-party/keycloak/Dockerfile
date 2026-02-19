# ==========================================================================================
# Author: Pablo González García.
# Created: 19/02/2026
# Last edited: 19/02/2026
# ==========================================================================================

# ==============================
# IMAGE
# ==============================

FROM quay.io/keycloak/keycloak:24.0

# Build-time options: these are baked into the optimized image
# and cannot be changed at runtime.
ENV KC_DB=postgres
ENV KC_HEALTH_ENABLED=true
ENV KC_METRICS_ENABLED=true
ENV KC_HOSTNAME_STRICT=false

RUN /opt/keycloak/bin/kc.sh build