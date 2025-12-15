# ==========================================================================================
# Author: Pablo González García.
# Created: 13/12/2025
# Last edited: 13/12/2025
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
            "api.main:app",
            host="0.0.0.0",
            port=9000
        )
    
    # If an unexpected error ocurred.
    except Exception as ex:
        # Prints information.
        print(f"Fatal error: {ex}")