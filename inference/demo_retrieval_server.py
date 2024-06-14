# Copyright 2020 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python AsyncIO implementation of the GRPC retrieval.Greeter server."""

import asyncio
import logging
import time
import numpy as np

import grpc
import retrieval_pb2
import retrieval_pb2_grpc


class RetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    async def Retrieve(
        self,
        request: retrieval_pb2.RetrievalRequest,
        context: grpc.aio.ServicerContext,
    ) -> retrieval_pb2.RetrievalReply:

        chunk_size = 64
        retrieved_tokens = []
        # for i, query in enumerate(request.query):
        #     print(f"Query {i}: {query}")
            # retrieved_tokens.append("Hello, {}!, num_continuation_chunks: {}, num_neighbours: {}, staleness_offset: {}  seq_len: {}, interval: {}, ".format(
            #     query, request.num_continuation_chunks, request.num_neighbours, request.staleness_offset, request.seq_len, request.interval))
        retrieved_tokens = list(np.ones(len(request.query) * 1 * request.num_neighbours * (1 + request.num_continuation_chunks) * chunk_size).astype(np.int32))
        print("batch size: {}\tnum_continuation_chunks: {}\tnum_neighbours: {}\tstaleness_offset: {}\tseq_len: {}\tinterval: {}\tuse_perf_model: {}".format(
            len(request.query), request.num_continuation_chunks, request.num_neighbours, request.staleness_offset, request.seq_len, request.interval, request.use_perf_model))
        print("Length of retrieved tokens: ", len(retrieved_tokens))
        time.sleep(0.1)
        return retrieval_pb2.RetrievalReply(retrieved_tokens=retrieved_tokens)
    
    def SetNprobe(
            self, 
            request: retrieval_pb2.SetNprobeRequest,
            context : grpc.ServicerContext
        ) -> retrieval_pb2.SetNprobeReply:
        print("SetNprobe: ", request.nprobe)
        return retrieval_pb2.SetNprobeReply(reply="Set nprobe to {} succeed".format(request.nprobe))


async def serve() -> None:
    server = grpc.aio.server()
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalService(), server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())