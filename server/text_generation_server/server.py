import asyncio
import os
import re
import torch
import time

from grpc import aio, insecure_channel
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from text_generation_server.cache import Cache
from text_generation_server.interceptor import ExceptionInterceptor
from text_generation_server.models import Model, get_model
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.tracing import UDSOpenTelemetryAioServerInterceptor, TCPOpenTelemetryAioServerInterceptor
from text_generation_server.models.idefics_causal_lm import IdeficsCausalLMBatch


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(
        self,
        model: Model,
        cache: Cache,
        quantize: Optional[str],
        server_urls: List[str],
        world_size: int
    ):
        self.cache = cache
        self.model = model
        self.quantize = quantize
        self.server_urls = set(server_urls)
        self.world_size = world_size
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of TextGenerationService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return generate_pb2.HealthResponse(
            world_size=self.world_size,
            registered_shards=len(self.server_urls),
            all_shards_registered=(self.world_size == len(self.server_urls))
        )

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=list(self.server_urls))

    async def ServiceRegistration(self, request, context):
        other_server_urls = self.server_urls.difference(set(request.urls))
        self.server_urls = self.server_urls.union(set(request.urls))
        return generate_pb2.ServiceRegistrationResponse(urls=list(other_server_urls))

    async def ClearCache(self, request, context):
        if request.HasField("id"):
            self.cache.delete(request.id)
        else:
            self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    async def FilterBatch(self, request, context):
        batch = self.cache.pop(request.batch_id)
        if batch is None:
            raise ValueError(f"Batch ID {request.batch_id if request is not None else None} not found in cache.")
        filtered_batch = batch.filter(request.request_ids)
        logger.info(f"FilterBatch cache.set({filtered_batch.batch_id if filtered_batch is not None else None})")
        self.cache.set(filtered_batch)

        return generate_pb2.FilterBatchResponse(batch=filtered_batch.to_pb())

    async def Warmup(self, request, context):
        if self.quantize == "gptq":
            try:
                # When using GPTQ, Exllama kernels need some global kernels
                # For which we have the finale shapes only after the model has loaded
                # This will allocate those buffers.
                from text_generation_server.utils.layers import (
                    create_exllama_buffers,
                    set_device,
                )

                set_device(self.model.device)
                create_exllama_buffers(request.max_prefill_tokens)
            except ImportError:
                pass

        if (
            self.model.batch_type == IdeficsCausalLMBatch
        ):  # Hack, i would rather use kwargs in the `from_pb` call
            batch = self.model.batch_type.from_pb(
                request.batch,
                self.model.tokenizer,
                self.model.processor,
                self.model.dtype,
                self.model.device,
            )
        else:
            batch = self.model.batch_type.from_pb(
                request.batch, self.model.tokenizer, self.model.dtype, self.model.device
            )
        max_supported_total_tokens = self.model.warmup(batch)

        return generate_pb2.WarmupResponse(
            max_supported_total_tokens=max_supported_total_tokens
        )

    async def Prefill(self, request, context):
        start = time.time_ns()
        if (
            self.model.batch_type == IdeficsCausalLMBatch
        ):  # Hack, i would rather use kwargs in the `from_pb` call
            batch = self.model.batch_type.from_pb(
                request.batch,
                self.model.tokenizer,
                self.model.processor,
                self.model.dtype,
                self.model.device,
            )
        else:
            batch = self.model.batch_type.from_pb(
                request.batch, self.model.tokenizer, self.model.dtype, self.model.device
            )

        generations, next_batch, timings = self.model.generate_token(batch)
        logger.info(f"Prefill cache.set({next_batch.batch_id if next_batch is not None else None})")
        self.cache.set(next_batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )

    async def Decode(self, request, context):
        start = time.time_ns()
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        batches = []
        for batch_pb in request.batches:
            batch = self.cache.pop(batch_pb.id)
            if batch is None:
                raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
            batches.append(batch)

        if len(batches) == 0:
            raise ValueError("All batches are empty")

        if len(batches) > 1:
            start_concat = time.time_ns()
            batch = self.model.batch_type.concatenate(batches)
            concat_ns = time.time_ns() - start_concat
        else:
            batch = batches[0]
            concat_ns = None

        generations, next_batch, timings = self.model.generate_token(batch)
        logger.info(f"Decode cache.set({next_batch.batch_id if next_batch is not None else None})")
        self.cache.set(next_batch)

        return generate_pb2.DecodeResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            concat_ns=concat_ns,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )


def _format_shard_uri(prefix:str, rank:int):
    m = re.match(r"tcp://([^:]*):(\d+)", prefix)
    if m is not None:
        return f"tcp://{m.group(1)}:{int(m.group(2))+rank}"
    return f"{prefix}-{rank}"


def serve(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[str],
    trust_remote_code: bool,
    shard_uri_prefix: str,
):
    async def serve_inner(
        model_id: str,
        revision: Optional[str],
        sharded: bool = False,
        quantize: Optional[str] = None,
        speculate: Optional[int] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if sharded:
            server_uris = [
                _format_shard_uri(shard_uri_prefix, rank)
                for rank in range(int(os.environ["NUM_SHARD"]))
            ]
            local_uri = server_uris[int(os.environ["RANK_LOCAL"])]
        else:
            local_uri = _format_shard_uri(shard_uri_prefix, 0)
            server_uris = [local_uri]

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        distributed_world = int(os.getenv("NUM_SHARD", "1")) < world_size
        master_rank = int(os.getenv("RANK", "0")) == 0

        if distributed_world:
            os.environ["INIT_METHOD"] = os.getenv(
                "INIT_METHOD",
                f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
            )

        try:
            model = get_model(
                model_id,
                revision,
                sharded or distributed_world,
                quantize,
                speculate,
                dtype,
                trust_remote_code,
            )
        except Exception:
            logger.exception("Error when initializing model")
            raise

        # Can only do this after the distributed model has connected to the master shard
        if (distributed_world and not master_rank):
            assert (
                shard_uri_prefix.startswith("tcp://")
            ), "Local shard URI must be TCP when NUM_SHARD < WORLD_SIZE"
            assert (
                os.getenv("MASTER_ADDR", None) is not None
            ), "MASTER_ADDR must be set when NUM_SHARD < WORLD_SIZE"
            assert (
                os.getenv("WORLD_SERVICE_PORT", None) is not None
            ), "WORLD_SERVICE_PORT must be set for non-master shards when NUM_SHARD < WORLD_SIZE"

            channel_url = f"{os.environ['MASTER_ADDR']}:{os.environ['WORLD_SERVICE_PORT']}"
            stub = generate_pb2_grpc.TextGenerationServiceStub(
                insecure_channel(channel_url)
            )

            deadline = time.time() + int(os.getenv("REGISTRATION_TIMEOUT_SECONDS", "60"))
            while True:
                try:
                    stub.Health(
                        generate_pb2.HealthRequest()
                    ) 
                    stub.ServiceRegistration(
                        generate_pb2.ServiceRegistrationRequest(
                            urls=server_uris
                        )
                    )
                    break
                except BaseException as err:
                    if time.time() >= deadline:
                        raise RuntimeError(f"Could not register with the master shard at {channel_url}") from err
                    time.sleep(0.1)

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                TCPOpenTelemetryAioServerInterceptor()
                if shard_uri_prefix.startswith("tcp://")
                else UDSOpenTelemetryAioServerInterceptor()
            ]
        )
        generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
            TextGenerationService(
                model,
                Cache(),
                quantize,
                server_uris,
                world_size
            ),
                server
        )
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["TextGenerationService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(
            local_uri[len("tcp://"):]
            if local_uri.startswith("tcp://")
            else local_uri
        )

        await server.start()

        logger.info("Server started at {}".format(local_uri))

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(
        serve_inner(
            model_id, revision, sharded, quantize, speculate, dtype, trust_remote_code
        )
    )
