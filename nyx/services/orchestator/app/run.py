# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from pathlib import Path

# External:
import uvicorn
# Internal:
from core.config import logging
from api.ws import dependencies


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Try-Except to manage errors.
    try:
        # Resolves absolute path.
        base_dir:Path = Path(__file__).resolve().parents[1]
        config_path:Path = base_dir / "config"

        # Initializes logger config.
        logging.setup_logging(config_path=config_path)

        # Initializes the dependencies.
        dependencies.setup_dependencies(config_path=config_path)

        # Run uvicorn.
        uvicorn.run(
            "api.ws.main:app",
            host="0.0.0.0",
            port=8000,
            ws_ping_interval=15,
            ws_ping_timeout=30,
            log_config=None
        )
    
    # If an unexpected error occurred.
    except Exception as ex:
        # Prints information.
        print(f"Fatal error: {ex}\n{ex.with_traceback(ex.__traceback__)}")