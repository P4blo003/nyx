# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
import uvicorn


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Try-Except to manage errors.
    try:
        # Run uvicorn.
        uvicorn.run(
            "api.ws.main:app",
            host="0.0.0.0",
            port=8000,
            ws_ping_interval=15,
            ws_ping_timeout=30
        )
    
    # If an unexpected error ocurred.
    except Exception as ex:
        # Prints information.
        print(f"Fatal error: {ex}")