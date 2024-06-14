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
"""The Python AsyncIO implementation of the GRPC helloworld.Greeter client."""

import asyncio
import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc

import time

async def run() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)

        start = time.time()
        request = stub.SayHello(helloworld_pb2.HelloRequest(name="you"))
        print("Do something else after sending request")
        response = await request
        end = time.time()
        print("Time: {:.2f} ms".format((end - start) * 1000))

        # response = await stub.SayHello(helloworld_pb2.HelloRequest(name="you"))
    print("Greeter client received: " + response.message)


if __name__ == "__main__":
    logging.basicConfig()
    asyncio.run(run())