# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from logging import Logger
from pathlib import Path

# External:
import uvicorn
# Internal:
from core.config import logger


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Try-Except to manage errors.
    try:
        # Resolves absolute path.
        base_dir:Path = Path(__file__).resolve().parents[1]
        config_path:Path = base_dir / "config" / "logging.yml"

        # Initializes logger config.
        logger.setup_logging(config_path=config_path)

        # Run uvicorn.
        uvicorn.run(
            "api.ws.main:app",
            host="0.0.0.0",
            port=8000,
            ws_ping_interval=15,
            ws_ping_timeout=30,
            log_config=None
        )
    
    # If an unexpected error ocurred.
    except Exception as ex:
        # Prints information.
        print(f"Fatal error: {ex}\n{ex.with_traceback(ex.__traceback__)}")