# ==========================================================================================
# Author: Pablo González García.
# Created: 10/02/2026
# Last edited: 10/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import time
import logging
import asyncio
from datetime import datetime
from argparse import ArgumentParser, Namespace

# External:
import grpc

# Internal:
import infrastructure.grpc.generated.ai_service_pb2 as pb2
import infrastructure.grpc.generated.ai_service_pb2_grpc as pb2_grpc
from shared.utilities import logging as logging_config


# ==============================
# CONFIGURATION
# ==============================

SERVER_HOST:str = "localhost"
SERVER_PORT:int = 8002

NUM_CLIENTS:int = 1
REQUESTS_PER_CLIENT:int = 10


# ==============================
# FUNCTIONS
# ==============================

def parse_args() -> Namespace:
    """
    """

    parser:ArgumentParser = ArgumentParser(description="AI Service Test")

    parser.add_argument("--server-host", type=str, default=SERVER_HOST, help="gRPC server host")
    parser.add_argument("--server-port", type=int, default=SERVER_PORT, help="gRPC server port")
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS, help="Number of concurrent clients to simulate")
    parser.add_argument("--requests-per-client", type=int, default=REQUESTS_PER_CLIENT, help="Number of requests each client will send")

    return parser.parse_args()

async def send_request(stub:pb2_grpc.AIServiceStub, request_id:int) -> None:
    """
    """

    return

async def client(id:int, stub:pb2_grpc.AIServiceStub, num_requests:int) -> None:
    """
    """

    log:logging.Logger = logging.getLogger(f"client-{id}")

    log.info(f"Starting client. Request to send: {num_requests}")
    
    start = time.perf_counter()

    for i in range(num_requests):

        req_start = time.perf_counter()

        await send_request(stub, i)

        req_duration = time.perf_counter() - start

        log.info(f"Request ({i}) processed - Elapsed time: {req_duration:.4f}s")

    duration =  time.perf_counter() - start

    log.info(f"Iteration completed - Elapsed time: {duration:.4f}s | Req/s: {(num_requests / duration):.4f}")

async def main(args:Namespace) -> None:
    """
    """

    log:logging.Logger = logging.getLogger("test")

    log.info(f"Starting test for service at {args.server_host}:{args.server_port}. ")

    async with grpc.aio.insecure_channel(f"{args.server_host}:{args.server_port}") as channel:

        stub:pb2_grpc.AIServiceStub = pb2_grpc.AIServiceStub(channel)

        start = time.perf_counter()

        # Load the embedding model.
        await stub.load_model(request=pb2.LoadModelRequest(
            server="triton-onnx",
            name="bge_m3_ensemble",
            version=""
        ))

        duration = time.perf_counter() - start
        
        log.info(f"Embedding model loaded - Elapsed time: {duration:.4f}s")

        # Simulate multiple concurrent clients.
        tasks = [client(index, stub, args.requests_per_client) for index in range(args.num_clients)]
        await asyncio.gather(*tasks)

        start = time.perf_counter()

        # Unloads the embedding model.
        await stub.unload_model(request=pb2.UnloadModelRequest(
            server="triton-onnx",
            name="bge_m3_ensemble",
            version=""
        ))

        duration = time.perf_counter() - start
        
        log.info(f"Embedding model unloaded - Elapsed time: {duration:.4f}s")


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    logging_config.setup(config=logging_config.get_config(
        log_level="INFO",
        log_dir="./logs/tests",
        app_name=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ))

    asyncio.run(main(parse_args()))