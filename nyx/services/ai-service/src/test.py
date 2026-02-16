# ==========================================================================================
# Author: Pablo González García.
# Created: 10/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import csv
import sys
import time
import math
import asyncio
import logging
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
from typing import List, Dict, Tuple
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

TASK_NAME:str = "embedding-inference"
MODEL_NAME:str = "bge_m3_ensemble"
MODEL_SERVER:str = "triton-onnx"

WARMUP_REQUESTS:int = 5
COOLDOWN_SECONDS:float = 2.0
REQUEST_TIMEOUT:float = 60.0


# ==============================
# LOGGING
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
log:logging.Logger = logging.getLogger("test")


# ==============================
# CLASSES
# ==============================

@dataclass
class RequestResult:
    """
    Result of a single inference request.
    """

    latency:float
    success:bool
    batch_size:int
    num_vectors:int = 0
    embedding_dim:int = 0
    error_code:Optional[str] = None
    error_message:Optional[str] = None


class Metrics:
    """
    Thread-safe metrics collector for a single test phase.
    """

    # ---- Default ---- #

    def __init__(self) -> None:
        """
        """

        # Initializes the class properties.
        self._lock:asyncio.Lock = asyncio.Lock()
        self._results:List[RequestResult] = []

        self._wall_start:Optional[float] = None
        self._wall_end:Optional[float] = None


    # ---- Methods ---- #

    def mark_start(self) -> None:
        """
        """

        self._wall_start = time.perf_counter()

    def mark_end(self) -> None:
        """
        """

        self._wall_end = time.perf_counter()

    @property
    def wall_time(self) -> float:
        """
        Total wall-clock time for the test phase.
        """

        if self._wall_start is None or self._wall_end is None: return 0.0
        return self._wall_end - self._wall_start

    async def record(self, result:RequestResult) -> None:
        """
        """

        async with self._lock:
            self._results.append(result)

    def compute_stats(self) -> Dict[str, Any]:
        """
        Compute latency and throughput statistics from recorded results.
        """

        successes:List[float] = sorted([r.latency for r in self._results if r.success])
        errors:List[RequestResult] = [r for r in self._results if not r.success]
        total:int = len(self._results)

        if total == 0: return {"total": 0}

        stats:Dict[str, Any] = {
            "total": total,
            "ok": len(successes),
            "errors": len(errors),
            "error_rate": len(errors) / total,
        }

        if successes:
            n:int = len(successes)
            stats.update({
                "avg": statistics.mean(successes),
                "median": statistics.median(successes),
                "stdev": statistics.stdev(successes) if n > 1 else 0.0,
                "p90": successes[min(int(n * 0.90), n - 1)],
                "p95": successes[min(int(n * 0.95), n - 1)],
                "p99": successes[min(int(n * 0.99), n - 1)],
                "min": successes[0],
                "max": successes[-1],
            })

        if self.wall_time > 0:
            stats["throughput_rps"] = len(successes) / self.wall_time

        if errors:
            error_summary:Dict[str, int] = {}
            for e in errors:
                key:str = e.error_code or "UNKNOWN"
                error_summary[key] = error_summary.get(key, 0) + 1
            stats["error_breakdown"] = error_summary

        return stats


class TextPool:
    """
    Provides cyclic text batches for concurrent workers without contention.
    Each worker gets its own offset to avoid lock overhead.
    """

    # ---- Default ---- #

    def __init__(self, texts:List[str]) -> None:
        """
        """

        # Initializes the class properties.
        self._texts:List[str] = texts

    # ---- Methods ---- #

    def get_single(self, index:int) -> str:
        """
        """

        return self._texts[index % len(self._texts)]

    def get_batch(self, start_index:int, n:int) -> List[str]:
        """
        """

        return [self._texts[(start_index + i) % len(self._texts)] for i in range(n)]


# ==============================
# FUNCTIONS
# ==============================

# ---- Arguments ---- #

def parse_args() -> Namespace:
    """
    """

    parser:ArgumentParser = ArgumentParser(
        description="AI Service load and correctness test for embedding inference."
    )

    # Server.
    parser.add_argument("--server-host", type=str, default=SERVER_HOST, help="gRPC server host.")
    parser.add_argument("--server-port", type=int, default=SERVER_PORT, help="gRPC server port.")

    # Model.
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Inference task name.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model to load/unload in Triton.")
    parser.add_argument("--model-server", type=str, default=MODEL_SERVER, help="Triton server identifier.")

    # Load profile.
    parser.add_argument("--clients-step", type=int, default=10, help="Client increment per step.")
    parser.add_argument("--clients-max", type=int, default=100, help="Maximum number of concurrent clients.")
    parser.add_argument("--requests-per-client", type=int, default=50, help="Number of requests each client sends per step.")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of texts per inference request.")
    parser.add_argument("--think-time", type=float, default=0.0, help="Delay in seconds between consecutive requests per worker.")

    # Test control.
    parser.add_argument("--warmup-requests", type=int, default=WARMUP_REQUESTS, help="Warmup requests before actual test (excluded from metrics).")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SECONDS, help="Cooldown in seconds between load increments.")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT, help="Timeout in seconds for each gRPC request.")

    # Input / Output.
    parser.add_argument("--texts-file", type=Path, required=True, help="Path to file with input texts (one per line).")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"), help="Directory for output CSV reports.")

    # Validation.
    parser.add_argument("--expected-dim", type=int, default=0, help="Expected embedding dimension. 0 to skip validation.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip response correctness validation entirely.")

    return parser.parse_args()


# ---- IO ---- #

def load_texts(path:Path) -> List[str]:
    """
    Load input texts from a file, one text per line.
    """

    if not path.exists():
        raise FileNotFoundError(f"Texts file not found: {path}")

    texts:List[str] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not texts:
        raise ValueError(f"Texts file is empty: {path}")

    return texts


def write_csv(path:Path, rows:List[Dict[str, Any]]) -> None:
    """
    Write all result rows to a CSV file.
    """

    if not rows: return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer:csv.DictWriter = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Results written to {path}")


# ---- Validation ---- #

def validate_response(
    response:pb2.InferenceResponse,
    batch_size:int,
    expected_dim:int
) -> Tuple[bool, str]:
    """
    Validate that the inference response is structurally correct.

    Returns:
        Tuple of (is_valid, error_message).
    """

    # Must contain an embedding batch.
    if not response.HasField("embedding_batch"):
        return False, "Response missing 'embedding_batch' field."

    batch:pb2.EmbeddingBatch = response.embedding_batch
    num_vectors:int = len(batch.vectors)

    # Number of output vectors must match the input batch size.
    if num_vectors != batch_size:
        return False, f"Expected {batch_size} vectors, got {num_vectors}."

    # Each vector must have the same dimension.
    if num_vectors > 0:
        first_dim:int = len(batch.vectors[0].values)

        if first_dim == 0:
            return False, "First embedding vector has 0 dimensions."

        if expected_dim > 0 and first_dim != expected_dim:
            return False, f"Expected dimension {expected_dim}, got {first_dim}."

        for i, vec in enumerate(batch.vectors):
            if len(vec.values) != first_dim:
                return False, f"Vector {i} has dimension {len(vec.values)}, expected {first_dim}."

            # Check for all-zero vectors (degenerate embeddings).
            if all(v == 0.0 for v in vec.values):
                return False, f"Vector {i} is all zeros."

            # Check for NaN / Inf values.
            for j, v in enumerate(vec.values):
                if math.isnan(v) or math.isinf(v):
                    return False, f"Vector {i} contains NaN/Inf at index {j}."

    return True, ""


# ---- Transport ---- #

async def send_request(
    stub:pb2_grpc.AIServiceStub,
    texts:List[str],
    task:str,
    timeout:float,
    validate:bool,
    expected_dim:int
) -> RequestResult:
    """
    Send a single inference request and return the measured result.
    """

    batch_size:int = len(texts)

    try:

        request:pb2.InferenceRequest = pb2.InferenceRequest(
            task=task,
            text_batch=pb2.TextBatch(content=texts)
        )

        start:float = time.perf_counter()
        response:pb2.InferenceResponse = await stub.make_infer(request=request, timeout=timeout)
        latency:float = time.perf_counter() - start

        # Validate response correctness.
        if validate:
            is_valid, error_msg = validate_response(
                response=response,
                batch_size=batch_size,
                expected_dim=expected_dim
            )

            if not is_valid:
                return RequestResult(
                    latency=latency,
                    success=False,
                    batch_size=batch_size,
                    error_code="VALIDATION_FAILED",
                    error_message=error_msg
                )

        # Extract response metadata.
        num_vectors:int = len(response.embedding_batch.vectors) if response.HasField("embedding_batch") else 0
        embedding_dim:int = len(response.embedding_batch.vectors[0].values) if num_vectors > 0 else 0

        return RequestResult(
            latency=latency,
            success=True,
            batch_size=batch_size,
            num_vectors=num_vectors,
            embedding_dim=embedding_dim
        )

    except grpc.aio.AioRpcError as rpc_err:

        return RequestResult(
            latency=0.0,
            success=False,
            batch_size=batch_size,
            error_code=rpc_err.code().name,
            error_message=rpc_err.details()
        )

    except Exception as ex:

        return RequestResult(
            latency=0.0,
            success=False,
            batch_size=batch_size,
            error_code=type(ex).__name__,
            error_message=str(ex)
        )


async def worker(
    worker_id:int,
    stub:pb2_grpc.AIServiceStub,
    pool:TextPool,
    metrics:Metrics,
    num_requests:int,
    batch_size:int,
    task:str,
    timeout:float,
    think_time:float,
    validate:bool,
    expected_dim:int
) -> None:
    """
    Simulates a single client sending sequential requests.
    """

    offset:int = worker_id * num_requests * batch_size

    for i in range(num_requests):
        texts:List[str] = pool.get_batch(start_index=offset + (i * batch_size), n=batch_size)

        result:RequestResult = await send_request(
            stub=stub,
            texts=texts,
            task=task,
            timeout=timeout,
            validate=validate,
            expected_dim=expected_dim
        )

        await metrics.record(result=result)

        if think_time > 0 and i < num_requests - 1:
            await asyncio.sleep(think_time)


# ---- Test phases ---- #

async def check_connectivity(stub:pb2_grpc.AIServiceStub, args:Namespace) -> bool:
    """
    Verify the gRPC connection to the AI Service is alive.
    """

    log.info("Checking server connectivity ...")

    try:
        # Try listing models as a connectivity check.
        await stub.list_models(
            request=pb2.ListModelsRequest(filter=""),
            timeout=10.0
        )
        log.info("Server is reachable.")
        return True

    except grpc.aio.AioRpcError as e:
        # UNIMPLEMENTED is acceptable - server is alive but RPC not implemented.
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            log.warning("list_models RPC not implemented, but server is reachable.")
            return True
        log.error(f"Server connectivity check failed: [{e.code().name}] {e.details()}")
        return False

    except Exception as e:
        log.error(f"Server connectivity check failed: {e}")
        return False


async def load_model(stub:pb2_grpc.AIServiceStub, args:Namespace) -> bool:
    """
    Load the model in Triton and verify it's READY.
    """

    log.info(f"Loading model '{args.model_name}' on server '{args.model_server}' ...")

    try:
        start:float = time.perf_counter()

        response:pb2.ModelStatus = await stub.load_model(
            request=pb2.LoadModelRequest(
                server=args.model_server,
                name=args.model_name,
                version=""
            ),
            timeout=120.0
        )

        duration:float = time.perf_counter() - start

        if response.state != pb2.ModelStatus.READY:
            log.error(f"Model loaded but state is {pb2.ModelStatus.State.Name(response.state)}: {response.message}")
            return False

        log.info(f"Model loaded successfully in {duration:.2f}s (state={pb2.ModelStatus.State.Name(response.state)}).")
        return True

    except grpc.aio.AioRpcError as e:
        log.error(f"Failed to load model: [{e.code().name}] {e.details()}")
        return False


async def unload_model(stub:pb2_grpc.AIServiceStub, args:Namespace) -> bool:
    """
    Unload the model from Triton.
    """

    log.info(f"Unloading model '{args.model_name}' ...")

    try:
        start:float = time.perf_counter()

        response:pb2.ModelStatus = await stub.unload_model(
            request=pb2.UnloadModelRequest(
                server=args.model_server,
                name=args.model_name,
                version=""
            ),
            timeout=60.0
        )

        duration:float = time.perf_counter() - start
        log.info(f"Model unloaded in {duration:.2f}s (state={pb2.ModelStatus.State.Name(response.state)}).")
        return True

    except grpc.aio.AioRpcError as e:
        log.error(f"Failed to unload model: [{e.code().name}] {e.details()}")
        return False


async def run_warmup(stub:pb2_grpc.AIServiceStub, pool:TextPool, args:Namespace) -> bool:
    """
    Send warmup requests to stabilize Triton's internal state (JIT, memory allocation, CUDA context).
    Results are discarded.
    """

    if args.warmup_requests <= 0: return True

    log.info(f"Running {args.warmup_requests} warmup request(s) (batch_size={args.batch_size}) ...")

    for i in range(args.warmup_requests):
        texts:List[str] = pool.get_batch(start_index=i * args.batch_size, n=args.batch_size)

        result:RequestResult = await send_request(
            stub=stub,
            texts=texts,
            task=args.task,
            timeout=args.timeout,
            validate=not args.skip_validation,
            expected_dim=args.expected_dim
        )

        if not result.success:
            log.error(f"Warmup request {i + 1} failed: [{result.error_code}] {result.error_message}")
            return False

        log.info(f"  Warmup {i + 1}/{args.warmup_requests}: {result.latency:.4f}s (dim={result.embedding_dim})")

    log.info("Warmup complete.")
    return True


async def run_load_step(
    channel:grpc.aio.Channel,
    pool:TextPool,
    args:Namespace,
    num_clients:int
) -> Metrics:
    """
    Run a single load step with a given number of concurrent clients.
    All workers share a single gRPC channel (as in production with connection pooling).
    """

    stub:pb2_grpc.AIServiceStub = pb2_grpc.AIServiceStub(channel=channel)
    metrics:Metrics = Metrics()

    metrics.mark_start()

    tasks:List[asyncio.Task] = [
        asyncio.create_task(worker(
            worker_id=i,
            stub=stub,
            pool=pool,
            metrics=metrics,
            num_requests=args.requests_per_client,
            batch_size=args.batch_size,
            task=args.task,
            timeout=args.timeout,
            think_time=args.think_time,
            validate=not args.skip_validation,
            expected_dim=args.expected_dim
        ))
        for i in range(num_clients)
    ]

    await asyncio.gather(*tasks)

    metrics.mark_end()

    return metrics


# ---- Reporting ---- #

def format_stat(value:Any, fmt:str = ".4f") -> str:
    """
    """

    if value is None: return "N/A"
    if isinstance(value, float): return f"{value:{fmt}}"
    return str(value)


def print_step_results(num_clients:int, stats:Dict[str, Any]) -> None:
    """
    """

    log.info(f"  Results for {num_clients} concurrent client(s):")
    log.info(f"    Requests:    {stats.get('ok', 0)} OK / {stats.get('errors', 0)} ERR  (total: {stats.get('total', 0)})")
    log.info(f"    Error rate:  {stats.get('error_rate', 0):.2%}")
    log.info(f"    Throughput:  {format_stat(stats.get('throughput_rps'), '.2f')} req/s")
    log.info(f"    Latency avg: {format_stat(stats.get('avg'))} s")
    log.info(f"    Latency med: {format_stat(stats.get('median'))} s")
    log.info(f"    Latency p90: {format_stat(stats.get('p90'))} s")
    log.info(f"    Latency p95: {format_stat(stats.get('p95'))} s")
    log.info(f"    Latency p99: {format_stat(stats.get('p99'))} s")
    log.info(f"    Latency min: {format_stat(stats.get('min'))} s")
    log.info(f"    Latency max: {format_stat(stats.get('max'))} s")
    log.info(f"    Stdev:       {format_stat(stats.get('stdev'))} s")

    if "error_breakdown" in stats:
        for code, count in stats["error_breakdown"].items():
            log.info(f"    Error [{code}]: {count}")


def build_csv_rows(
    all_stats:Dict[int, Dict[str, Any]],
    args:Namespace
) -> List[Dict[str, Any]]:
    """
    Build CSV rows from all load step statistics.
    """

    rows:List[Dict[str, Any]] = []

    for num_clients, stats in all_stats.items():
        rows.append({
            "clients": num_clients,
            "batch_size": args.batch_size,
            "requests_per_client": args.requests_per_client,
            "think_time_s": args.think_time,
            "total_requests": stats.get("total", 0),
            "ok_requests": stats.get("ok", 0),
            "errors": stats.get("errors", 0),
            "error_rate": round(stats.get("error_rate", 0), 4),
            "throughput_rps": round(stats.get("throughput_rps", 0), 2),
            "latency_avg_s": round(stats.get("avg", 0), 6),
            "latency_median_s": round(stats.get("median", 0), 6),
            "latency_stdev_s": round(stats.get("stdev", 0), 6),
            "latency_p90_s": round(stats.get("p90", 0), 6),
            "latency_p95_s": round(stats.get("p95", 0), 6),
            "latency_p99_s": round(stats.get("p99", 0), 6),
            "latency_min_s": round(stats.get("min", 0), 6),
            "latency_max_s": round(stats.get("max", 0), 6),
        })

    return rows


def print_summary(all_stats:Dict[int, Dict[str, Any]]) -> None:
    """
    """

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    total_requests:int = sum(s.get("total", 0) for s in all_stats.values())
    total_errors:int = sum(s.get("errors", 0) for s in all_stats.values())
    total_ok:int = sum(s.get("ok", 0) for s in all_stats.values())

    log.info(f"  Total requests sent:      {total_requests}")
    log.info(f"  Total successful:         {total_ok}")
    log.info(f"  Total errors:             {total_errors}")
    log.info(f"  Overall error rate:       {total_errors / total_requests:.2%}" if total_requests > 0 else "  Overall error rate:       N/A")

    # Find the peak throughput.
    if all_stats:
        peak_clients, peak_stats = max(
            all_stats.items(),
            key=lambda x: x[1].get("throughput_rps", 0)
        )
        log.info(f"  Peak throughput:          {peak_stats.get('throughput_rps', 0):.2f} req/s (at {peak_clients} clients)")

        # Find the step where p99 exceeds a threshold or errors spike.
        for num_clients, stats in all_stats.items():
            if stats.get("error_rate", 0) > 0.05:
                log.warning(f"  Degradation detected at {num_clients} clients: error rate {stats['error_rate']:.2%}")
                break

    log.info("=" * 60)


# ---- Main ---- #

async def main(args:Namespace) -> None:
    """
    """

    timestamp:str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load input texts.
    texts:List[str] = load_texts(path=args.texts_file)
    pool:TextPool = TextPool(texts=texts)

    # Compute client increments.
    increments:List[int] = list(range(args.clients_step, args.clients_max + 1, args.clients_step))
    if not increments or increments[-1] != args.clients_max:
        increments.append(args.clients_max)
    # Always start with 1 client as baseline.
    if increments[0] != 1:
        increments.insert(0, 1)

    total_requests:int = sum(c * args.requests_per_client for c in increments)

    # Print test plan.
    log.info("")
    log.info("=" * 60)
    log.info("AI SERVICE - LOAD & CORRECTNESS TEST")
    log.info("=" * 60)
    log.info(f"  Server:              {args.server_host}:{args.server_port}")
    log.info(f"  Task:                {args.task}")
    log.info(f"  Model:               {args.model_name} @ {args.model_server}")
    log.info(f"  Batch size:          {args.batch_size} texts/request")
    log.info(f"  Requests per client: {args.requests_per_client}")
    log.info(f"  Think time:          {args.think_time}s")
    log.info(f"  Client increments:   {increments}")
    log.info(f"  Total requests:      {total_requests}")
    log.info(f"  Warmup requests:     {args.warmup_requests}")
    log.info(f"  Cooldown:            {args.cooldown}s")
    log.info(f"  Request timeout:     {args.timeout}s")
    log.info(f"  Input texts:         {len(texts)}")
    log.info(f"  Validation:          {'disabled' if args.skip_validation else 'enabled'}")
    if args.expected_dim > 0:
        log.info(f"  Expected dimension:  {args.expected_dim}")
    log.info(f"  Output directory:    {args.output_dir}")
    log.info("=" * 60)
    log.info("")

    # Open a single long-lived channel for the entire test.
    async with grpc.aio.insecure_channel(
        f"{args.server_host}:{args.server_port}",
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]
    ) as channel:

        stub:pb2_grpc.AIServiceStub = pb2_grpc.AIServiceStub(channel=channel)

        # Phase 1: Connectivity.
        if not await check_connectivity(stub=stub, args=args):
            log.error("Aborting: server is not reachable.")
            sys.exit(1)

        # Phase 2: Load model.
        if not await load_model(stub=stub, args=args):
            log.error("Aborting: failed to load model.")
            sys.exit(1)

        # Phase 3: Warmup.
        if not await run_warmup(stub=stub, pool=pool, args=args):
            log.error("Aborting: warmup failed. The model may not be functioning correctly.")
            sys.exit(1)

        # Phase 4: Load test.
        log.info("")
        log.info("=" * 60)
        log.info("LOAD TEST")
        log.info("=" * 60)

        all_stats:Dict[int, Dict[str, Any]] = {}

        for step_index, num_clients in enumerate(increments):
            log.info("")
            log.info(f"Step {step_index + 1}/{len(increments)}: {num_clients} concurrent client(s) ...")

            metrics:Metrics = await run_load_step(
                channel=channel,
                pool=pool,
                args=args,
                num_clients=num_clients
            )

            stats:Dict[str, Any] = metrics.compute_stats()
            all_stats[num_clients] = stats

            print_step_results(num_clients=num_clients, stats=stats)

            # Cooldown between steps (except after the last one).
            if step_index < len(increments) - 1 and args.cooldown > 0:
                log.info(f"  Cooling down for {args.cooldown}s ...")
                await asyncio.sleep(args.cooldown)

        # Phase 5: Unload model.
        log.info("")
        await unload_model(stub=stub, args=args)

    # Phase 6: Report.
    print_summary(all_stats=all_stats)

    # Write CSV.
    csv_rows:List[Dict[str, Any]] = build_csv_rows(all_stats=all_stats, args=args)
    csv_filename:str = f"load_test-{args.model_name}-bs{args.batch_size}-{timestamp}.csv"
    csv_path:Path = args.output_dir / csv_filename
    write_csv(path=csv_path, rows=csv_rows)

    log.info("")
    log.info("Test complete.")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    asyncio.run(main(args=parse_args()))
