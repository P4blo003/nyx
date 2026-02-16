# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
import time
from typing import Any
from typing import Awaitable, Callable, AsyncIterable
from typing import Coroutine
from typing import cast

# Internal
import grpc
from grpc import HandlerCallDetails, RpcMethodHandler, aio


# ==============================
# CLASSES
# ==============================

class RequestLogInterceptor(aio.ServerInterceptor):
    """
    
    """
    
    # ---- Default ---- #

    def __init__(self, log:logging.Logger) -> None:
        """
        Initializes the interceptor.
        """

        # Initializes the class properties.
        self._log = log


    # ---- Methods ---- #

    async def intercept_service(
        self,
        continuation: Callable[[HandlerCallDetails],Awaitable[Any]],
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """
        
        """

        method_name:str = handler_call_details.method
        handler = await continuation(handler_call_details)

        if handler is None: return handler

        # If it's a simple request.
        if handler.unary_unary:
            original_behavior = cast(
                Callable[[Any, aio.ServicerContext], Coroutine[Any, Any, Any]], 
                handler.unary_unary
            )

            async def unary_wrapper(request:Any, context:aio.ServicerContext) -> Any:
                """"""
                
                start_time:float = time.perf_counter()
                
                try:

                    self._log.info(f"{context.peer()} > {method_name}")
                    return await original_behavior(request, context)
                
                finally:

                    duration:float = time.perf_counter() - start_time
                    self._log.info(f"{context.peer()} > {method_name} processed in {duration:.2f} seconds")

            return grpc.unary_unary_rpc_method_handler(
                unary_wrapper,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer
            )
        

        if handler.unary_stream:
            original_behavior = cast(
                Callable[[Any, aio.ServicerContext], AsyncIterable[Any]], 
                handler.unary_stream
            )

            async def stream_wrapper(request:Any, context:aio.ServicerContext) -> Any:
                """"""

                start_time:float = time.perf_counter()
                count:int = 0

                try:
                    self._log.info(f"{context.peer()} -> {method_name}")
                    async for response in original_behavior(request, context):
                        count += 1
                        yield response

                finally:

                    duration:float = time.perf_counter() - start_time
                    self._log.info(f"{context.peer()} -> {method_name} processed {count} streams in {duration:.2f} seconds")

            return grpc.unary_stream_rpc_method_handler(
                stream_wrapper,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer
            )
        
        return handler