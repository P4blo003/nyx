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
from core.config import setting


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
        # Initializes settings config.
        setting.setup_settings(config_path=config_path)

        # Checks if configuration was initialized.
        if setting.SERVICE_CONFIG is None: raise RuntimeError("Service Settings was not initialized.")

        # Run uvicorn.
        uvicorn.run(
            "api.main:app",
            host=setting.SERVICE_CONFIG.host,
            port=setting.SERVICE_CONFIG.port,
            ws_ping_interval=setting.SERVICE_CONFIG.ws_ping_interval,
            ws_ping_timeout=setting.SERVICE_CONFIG.ws_ping_timeout,
            log_config=None
        )
    
    # If an unexpected error occurred.
    except Exception as ex:
        # Prints information.
        print(f"Critical error: {ex}\n{ex.with_traceback(ex.__traceback__)}")