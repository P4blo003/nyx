# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# VARIABLES
# ==============================

# Start the qdrant service in detached mode.
# - Creates containers, networks, and volumes if they do not exist.
# - Does not rebuild images unless explicitly required.
up:
	docker compose up -d

# Stop and remove containers, but keep volumes and data.
# Useful when restarting services without losing persisted data.
down:
	docker compose down

# Stream logs from all services in real time.
# Helpful for debugging startup issues or runtime errors.
logs:
	docker compose logs -f

# Show the current status of containers managed by Docker Compose.
# Displays container state, ports and uptime.
status:
	docker compose ps

# Stop containers and remove associated volumes.
# WARNING: This will permanently delete persisted Qdrant data.
# Use only for full resets or non-production environments.
clean:
	docker compose down -v