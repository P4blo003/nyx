# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 18/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
from argparse import ArgumentParser, Namespace


# ==============================
# PROPERTIES
# ==============================

# Server configuration.
DEFAULT_HOST:str = "[::]"
DEFAULT_PORT:int = 8002


# ==============================
# FUNCTIONS
# ==============================

def parse_arguments() -> Namespace:
    """
    """

    parser:ArgumentParser = ArgumentParser()

    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=""
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=""
    )

    return parser.parse_args()

async def main(args:Namespace) -> None:
    """
    """

    pass


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Run main.
    asyncio.run(main=main(args=parse_arguments()))