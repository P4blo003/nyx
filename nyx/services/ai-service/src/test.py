# ==========================================================================================
# Author: Pablo González García.
# Created: 10/02/2026
# Last edited: 12/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import csv
import time
import random
import asyncio
import statistics
import itertools
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from typing import List, Dict
from typing import Any, Optional

# External:
import grpc

# Internal:
import infrastructure.grpc.generated.ai_service_pb2 as pb2
import infrastructure.grpc.generated.ai_service_pb2_grpc as pb2_grpc


# ==============================
# CONFIGURATION
# ==============================

SERVER_HOST:str = "localhost"
SERVER_PORT:int = 8002

CLIENTS_STEP:int = 10
CLIENTS_MAX:int = 100

CLIENTS_REQUESTS:int = 50

THINK_TIME:float = 0.05

NUM_TEXTS_IN_BATCH:int = 5


# ==============================
# CLASSES
# ==============================

class Metrics:
    """
    """

    # ---- Default ---- #

    def __init__(self) -> None:
        """
        
        """

        # Initializes the class properties.
        self._lock:asyncio.Lock = asyncio.Lock()

        self._latencies:List[float] = []
        self._errors:int = 0

        self._start:Optional[float] = None
        self._end:Optional[float] = None


    # ---- Methods ---- #

    async def record(self, latency:float) -> None:
        """
        """

        async with self._lock:
            self._latencies.append(latency)

    async def record_error(self) -> None:
        """
        """

        async with self._lock:
            self._errors += 1

    def compute_stats(self) -> Dict[str, Any]:
        """
        
        """

        lat_sorted:List[float] = sorted(self._latencies)
        total:int = len(lat_sorted)

        if total == 0: return {}

        return {
            "ok": len(self._latencies),
            "errors": self._errors,
            "avg": statistics.mean(lat_sorted),
            "p95": lat_sorted[int(total * 0.95) - 1],
            "p99": lat_sorted[int(total * 0.99) - 1],
            "min": lat_sorted[0],
            "max": lat_sorted[-1]
        }

class TextPool:
    """
    """

    # ---- Default ---- #

    def __init__(self, texts:List[str]) -> None:
        """
        
        """

        # Initializes the class properties.
        self._texts:List[str] = texts
        self._cycle:itertools.cycle = itertools.cycle(self._texts)
        self._lock:asyncio.Lock = asyncio.Lock()


    # ---- Methods ---- #

    async def next(self):
        """
        """

        async with self._lock:
            return next(self._cycle)
        
    async def batch(self, n:Optional[int] = None) -> List[str]:
        """
        """

        async with self._lock:
            if n is None: return self._texts
            return [next(self._cycle) for _ in range(n)]


# ==============================
# FUNCTIONS
# ==============================

# ---- Common ---- #

def parse_args() -> Namespace:
    """
    """

    parser:ArgumentParser = ArgumentParser(description="AI Service Test")

    parser.add_argument("--server-host", type=str, default=SERVER_HOST, help="gRPC server host")
    parser.add_argument("--server-port", type=int, default=SERVER_PORT, help="gRPC server port")
    
    parser.add_argument("--clients-step", type=int, default=CLIENTS_STEP, help="")
    parser.add_argument("--clients-max", type=int, default=CLIENTS_MAX, help="")
    parser.add_argument("--clients-requests", type=int, default=CLIENTS_REQUESTS, help="Number of requests each client will send")
    parser.add_argument("--single-request", action="store_true", help="")
    parser.add_argument("--num-texts-in-batch", type=int, default=NUM_TEXTS_IN_BATCH, help="")
    
    parser.add_argument("--think-time", type=float, default=THINK_TIME)

    parser.add_argument("--texts-file", type=Path, required=True, help="")

    return parser.parse_args()

def append_csv(path:str, row:Dict[str, Any]) -> None:
    """
    
    """

    write_header = not Path(path).exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter=",")

        if write_header:
            writer.writeheader()

        formatted_row = {k: f"{v:.4f}".replace(".", ",") if isinstance(v, float) else v for k, v in row.items()}
        writer.writerow(formatted_row)

def load_texts(path:str) -> List[str]:
    """
    
    """

    texts = Path(path).read_text(encoding="utf-8").splitlines()
    texts = [t.strip() for t in texts if t.strip()]

    if not texts:
        raise ValueError("Empty dataset")
    
    return texts


# ---- Transport ---- #

async def send_request(stub:pb2_grpc.AIServiceStub, metrics:Metrics, texts:List[str]) -> None:
    """
    """

    try:

        req:pb2.InferenceRequest = pb2.InferenceRequest(
            task="embedding-inference",
            text_batch=pb2.TextBatch(content=texts)
        )

        start:float = time.perf_counter()

        response = await stub.make_infer(request=req)

        duration:float = time.perf_counter() - start

        await metrics.record(latency=duration)

    except Exception as ex:

        await metrics.record_error()


async def worker(
    id:int,
    args:Namespace,
    metrics:Metrics,
    pool:TextPool
) -> None:
    """
    
    """

    async with grpc.aio.insecure_channel(f"{args.server_host}:{args.server_port}")as channel:

        stub:pb2_grpc.AIServiceStub = pb2_grpc.AIServiceStub(channel=channel)

        num_requests:int = args.clients_requests if args.single_request else args.clients_requests // args.num_texts_in_batch
        content_per_request:int = 1 if args.single_request else args.num_texts_in_batch

        print(f"Worker ({id})\n: ({num_requests}) Requests with {content_per_request} strings")

        if args.single_request:
            for _ in range(num_requests):
                await send_request(stub=stub, metrics=metrics, texts=[await pool.next()])
        else:
            for _ in range(num_requests):
                await send_request(stub=stub, metrics=metrics, texts=await pool.batch(n=content_per_request))
            
            if args.think_time > 0:
                await asyncio.sleep(args.think_time)


# ---- Main ---- #

async def main(args:Namespace) -> None:
    """
    
    """

    test_timestamp:str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    texts:List[str] = load_texts(path=args.texts_file)

    clients_increment:List[int] = list(range(1, args.clients_max + 1, args.clients_step))
    if clients_increment[-1] != args.clients_max: clients_increment.append(args.clients_max)

    num_increments:int = len(clients_increment)

    total_requests:int = sum(c * args.clients_requests for c in clients_increment)
    request_per_client:int = args.clients_requests if args.single_request else args.clients_requests // args.num_texts_in_batch
    content_per_request:int = 1 if args.single_request else min(args.num_texts_in_batch, len(texts))

    print("\n=== CONFIGURATION === \n")
    print(f"URL:                {args.server_host}:{args.server_port}.")
    print(f"CLIENTS STEP:       {args.clients_step}.")
    print(f"CLIENTS MAX:        {args.clients_max}.")
    print(f"CLIENT INCREMENTS:  {clients_increment}.")
    print(f"NUM INCREMENTS:     {num_increments}.")
    print(f"CLIENTS REQUESTS:   {request_per_client} req.")
    print(f"REQUEST_CONTENT:    {content_per_request} string/req.")
    print(f"TOTAL REQUESTS:     {total_requests} req.")
    print(f"SINGLE REQUEST:     {args.single_request}.")
    print(f"THINK TIME:         {args.think_time} s.")
    print(f"TEXTS SOURCE FILE:  {args.texts_file}.")
    print(f"NUM INPUTS:         {len(texts)}.")

    input("\nPress any key to continue.\n")

    pool:TextPool = TextPool(texts=texts)
    stats:Dict[int, Metrics] = {}


    async with grpc.aio.insecure_channel(f"{args.server_host}:{args.server_port}") as channel:
        
        stub:pb2_grpc.AIServiceStub = pb2_grpc.AIServiceStub(channel=channel)

        print(f"Loading model ...")

        start:float = time.perf_counter()

        await stub.load_model(request=pb2.LoadModelRequest(
            server="triton-onnx",
            name="bge_m3_ensemble",
            version=""
        ))

        duration:float = time.perf_counter()

        print(f"Model loaded in {duration:.4f} s.")

    input("\nPress any key to continue.\n")

    print("\n=== TESTS === \n")

    for i in range(num_increments):
        
        metrics:Metrics = Metrics()

        print(f"({i})\t[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Running increment with {clients_increment[i]} client(s) ...")

        start:float = time.perf_counter()

        tasks:List = [worker(id=index, args=args, metrics=metrics, pool=pool) for index in range(clients_increment[i])]
        await asyncio.gather(*tasks)

        duration:float = time.perf_counter() - start

        print(f"\tIncrement finished in {duration:.4f}s")

        stats[clients_increment[i]] = metrics

    
    async with grpc.aio.insecure_channel(f"{args.server_host}:{args.server_port}") as channel:

        stub:pb2_grpc.AIServiceStub = pb2_grpc.AIServiceStub(channel=channel)

        print(f"\nUnloading model ...")

        start:float = time.perf_counter()

        await stub.unload_model(request=pb2.UnloadModelRequest(
            server="triton-onnx",
            name="bge_m3_ensemble",
            version=""
        ))

        duration:float = time.perf_counter()

        print(f"\nModel unloaded in {duration:.4f} s.")


    print("\n=== RESULTS === \n")

    for num_clients, results in stats.items():

        results_dict:Dict[str, Any] = results.compute_stats()
        
        print(f"Stats for ({num_clients}) concurrent clients:")
        print(f"OK REQUESTS:        {results_dict["ok"] if results_dict.get("ok", None) is not None else "NaN"} req.")
        print(f"ERRORS:             {results_dict["errors"] if results_dict.get("errors", None) is not None else "NaN"}.")
        print(f"AVG:                {results_dict["avg"] if results_dict.get("avg", None) is not None else "NaN"} s.")
        print(f"P95:                {results_dict["p95"] if results_dict.get("p95", None) is not None else "NaN"} s.")
        print(f"P99:                {results_dict["p99"] if results_dict.get("p99", None) is not None else "NaN"} s.")
        print(f"MIN:                {results_dict["min"] if results_dict.get("min", None) is not None else "NaN"} s.")
        print(f"MAX:                {results_dict["max"] if results_dict.get("max", None) is not None else "NaN"} s.")

        row:Dict[str, Any] = {
            "clients": num_clients,
            "request_per_client": request_per_client,
            "content_per_request": content_per_request,
            "ok_requests": results_dict.get("ok", "NaN"),
            "errors": results_dict.get("errors", "NaN"),
            "latency_avg": results_dict.get("avg", "NaN"),
            "latency_p95": results_dict.get("p95", "NaN"),
            "latency_p99": results_dict.get("p99", "NaN"),
            "latency_min": results_dict.get("min", "NaN"),
            "latency_max": results_dict.get("max", "NaN")
        }

        csv_file:str = f"./results-{args.clients_max}_clients-{args.clients_step}_steps-"
        csv_file += "single_request" if args.single_request else "batched_request"
        csv_file += f"-{test_timestamp}.csv"
        
        append_csv(path=csv_file, row=row)

        input("\nPress any key to continue.\n")

    input("\nPress any key to exit.\n")


# ==============================
# FUNCTIONS
# ==============================

if __name__ == "__main__":

    asyncio.run(main(args=parse_args()))