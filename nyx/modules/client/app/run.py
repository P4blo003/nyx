# ==========================================================================================
# Author: Pablo González García.
# Created: 12/12/2025
# Last edited: 12/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
import asyncio
from argparse import Namespace, ArgumentParser
# External:
import websockets


# ==============================
# FUNCTIONS
# ==============================

def parse_args() -> Namespace:
    """
    Parse given arguments.

    Returns:
        Namespace: Class with given arguments.
    """
    # Creates parser.
    parser:ArgumentParser = ArgumentParser()

    # Add arguments to parse.
    parser.add_argument(
        "uri",
        type=str,
        help="Server uri to connect."
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of concurrent clients."
    )

    # Returns namespace with arguments.
    return parser.parse_args()

async def run_client(uri:str, id:int) -> None:
    """
    
    """
    # Try-Except to manage errors.
    try:
        # Open connection.
        async with websockets.connect(uri=uri) as ws:
            # Prints information.
            print(f"[Client {id}] connected to {uri}")
            # Awaits 10 seconds.
            await asyncio.sleep(delay=15)

    # If and unexpected error ocurred.
    except Exception as ex:
        # Prints information.
        print(f"Unexpected error: {ex}")

async def main() -> None:
    """
    Run x number of clients.
    """
    # Parse arguments.
    args:Namespace = parse_args()

    # Run tasks.
    await asyncio.gather(*(run_client(args.uri,i) for i in range(args.clients)))

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Run main function.
    asyncio.run(main())