# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
import sys
from datetime import datetime

# External:
import uvicorn

# Internal:
from interfaces.api import run as run_api


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    try:

        # Executes uvicorn.
        uvicorn.run(
            app=run_api.app,
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "80"))
        )

        sys.exit(0)
    
    # If an error occurs.
    except Exception as ex:
        
        # Prints the error.
        print(f"({datetime.now()}) [ERROR] => Critical error during main execution: {ex}")
        sys.exit(1)