# ==========================================================================================
# Author: Pablo González García.
# Created: 12/12/2025
# Last edited: 12/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
from argparse import Namespace, ArgumentParser
# External:
import websockets
# Internal:
from dto.base import MessageType
from dto.request import ClientRequest


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
        default=1,
        help="Number of concurrent clients."
    )

    # Returns namespace with arguments.
    return parser.parse_args()

async def run_client(uri:str, id:int) -> None:
    """
    Run client console interface.

    Args:
        uri (str): Web uri.
        id (int): Client identifier.
    """
    # Try-Except to manage errors.
    try:
        # Open connection.
        async with websockets.connect(uri=uri) as ws:
            # Prints information.
            print(f"[Client {id}] connected to {uri}")
            
            # Main loop to send queries.
            loop = asyncio.get_running_loop()
            while True:
                # Gets user input in a separate thread to avoid blocking the loop.
                query:str = await loop.run_in_executor(None, input, "🧑: ")

                # Creates the request.
                request:ClientRequest = ClientRequest(
                    mtype=MessageType.STREAM,
                    content=query
                )

                # Sends the message.
                await ws.send(message=request.model_dump_json())

                # Prints response.
                print(f"🤖: ", end="")
                # Waits for server response.
                while True:
                    # Gets message.
                    message = await ws.recv()

                    # Checks if message is a bytes object.
                    if isinstance(message, bytes):
                        # Decode message.
                        message = message.decode("utf-8")
                    
                    # Prints the message.
                    print(message, end="", flush=True)

                    # Break on end-of-stream or custom protocol.
                    if message.endswith("\n"):                      # type: ignore
                        break
                
                print()
    
    # If the task is cancelled.
    except asyncio.CancelledError:
        # Prints information.
        print(f"\n[Client {id}] Connection cancelled, shutting down.")
    
    # If and unexpected error occurred.
    except Exception as ex:
        # Prints information.
        print(f"\n[Client {id}] Unexpected error: {ex}")

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