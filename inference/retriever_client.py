"""
This is an example file showing the usage of retrievers. 

Example Usage:
    python retriever_client.py --host 127.0.0.1 --port 50051 --batch_size 1 --staleness 1 \
        --chunk_size 64 --num_continuation_chunks 1 --num_neighbours 1 --staleness_offset 0 --seq_len 1 --interval 64
"""

import asyncio
import numpy as np
import time

import grpc
import retrieval_pb2
import retrieval_pb2_grpc

from typing import Optional, List

class BaseRetriever:
    
    def __init__(self):
        raise NotImplementedError
    
    def retrieve(self):
        raise NotImplementedError
    
class DummyRetriever(BaseRetriever):

    def __init__(sel, chunk_size : Optional[int] = 64):
        self.chunk_size = chunk_size

    def retrieve(self, query : Optional = ['sample query'], 
        num_continuation_chunks : Optional[int] = 1, num_neighbours : Optional[int] = 1):
        # seq_len : Optional[int] = 1, 
        # interval : Optional[int] = 64, num_neighbours : Optional[int] = 1):
        """
        Return a list of results: 
            each with length of chunk_size * (1 + num_continuation_chunks) * num_neighbours 
        """
        batch_size = len(query)
        dummy_out = ["dummy " * (self.chunk_size * (1 + num_continuation_chunks) * num_neighbours)] * batch_size
            
        return dummy_out

class ExternalRetriever(BaseRetriever):

    def __init__(self, host: Optional[str] = None, port: Optional[int] = 50051, chunk_size : Optional[int] = 64):
          
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.request = None # outstanding request

    def create_channel(self):
        self.channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}") 
        self.stub = retrieval_pb2_grpc.RetrievalServiceStub(self.channel)

    # async def retrieve_send(self, query : Optional = ['sample query'], 
    #     num_continuation_chunks : Optional[int] = 1, num_neighbours : Optional[int] = 1,
    #     staleness_offset : Optional[int] = 0, seq_len : Optional[int] = 1, interval : Optional[int] = 64):
    #     """
    #     Send out a single query
    #     """
    #     self.create_channel()
    #     self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
    #         query=query, num_continuation_chunks=num_continuation_chunks, num_neighbours=num_neighbours, 
    #         staleness_offset=staleness_offset, seq_len=seq_len, interval=interval))
    #     response = await self.request
    #     retrieved_tokens = response.retrieved_tokens
    #     retrieved_tokens = response.retrieved_tokens
    #     print("RetrievalService client received: ", [str(r) for r in retrieved_tokens]) 
    
    # async def retrieve_recv(self):
    #     """
    #     Receive the answer to a single query, i.e., a list of results, len = batch_size
    #     """
    #     response = await self.request
    #     retrieved_tokens = response.retrieved_tokens
    #     print("RetrievalService client received: ", [str(r) for r in retrieved_tokens])
    #     return retrieved_tokens
    
    # def retrieve(self, query : Optional = ['sample query'], 
    #     num_continuation_chunks : Optional[int] = 1, num_neighbours : Optional[int] = 1,
    #     staleness_offset : Optional[int] = 0, seq_len : Optional[int] = 1, interval : Optional[int] = 64):
    #     """
    #     Send out query and receive the answer
    #     """
    #     self.create_channel()
    #     self.retrieve_send(query=query, num_continuation_chunks=num_continuation_chunks, num_neighbours=num_neighbours, 
    #         staleness_offset=staleness_offset, seq_len=seq_len, interval=interval)
    #     retrieved_tokens = self.retrieve_recv()
    #     # retrieved_tokens = None

    #     return retrieved_tokens

    async def retrieve(self, query : Optional = ['sample query'], 
        num_continuation_chunks : Optional[int] = 1, num_neighbours : Optional[int] = 1,
        staleness_offset : Optional[int] = 0, seq_len : Optional[int] = 1, interval : Optional[int] = 64):
        """
        Send out a single query
        """
        self.create_channel()
        self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
            query=query, num_continuation_chunks=num_continuation_chunks, num_neighbours=num_neighbours, 
            staleness_offset=staleness_offset, seq_len=seq_len, interval=interval))
        response = await self.request
        retrieved_tokens = response.retrieved_tokens
        print("RetrievalService client received: ", [str(r) for r in retrieved_tokens]) 
        return retrieved_tokens

    async def dummy_generation(self, staleness=True, query : Optional = ['sample query'], 
        num_continuation_chunks : Optional[int] = 1, num_neighbours : Optional[int] = 1,
        staleness_offset : Optional[int] = 0, seq_len : Optional[int] = 1, interval : Optional[int] = 64):
        """
        A example of a dummy inference function
        """
        print("Running dummy generation WITH staleness" if staleness else "Running dummy generation WITHOUT staleness")

        def empty_inference(retrieved_tokens):
            time.sleep(0.002)
        
        self.create_channel()

        total_steps = 512
        retrieved_tokens = [0] * len(query) * 1 * num_neighbours * (self.chunk_size * (1 + num_continuation_chunks))
        time_per_step = []

        if staleness:
            first_retrieval = True 
            has_unconsumed_retrieval = False
            response = None
            for i in range(total_steps):

                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, empty_inference, retrieved_tokens)
                
                if i > 0 and i % interval == 0:
                    # print("Step: ", i)
                    if first_retrieval:
                        # print("First retrieval")
                        self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
                            query=query, num_continuation_chunks=num_continuation_chunks, num_neighbours=num_neighbours, 
                            staleness_offset=staleness_offset, seq_len=seq_len, interval=interval))
                        first_retrieval = False
                        response = await self.request
                        has_unconsumed_retrieval = False
                        retrieved_tokens = response.retrieved_tokens
                    else:
                        # print("Continuation retrieval")
                        if has_unconsumed_retrieval:
                            # print("Using unconsumed retrieval")
                            response = await self.request
                            retrieved_tokens = response.retrieved_tokens
                        if i < total_steps - 1:
                            # print("Sending new retrieval")
                            self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
                                query=query, num_continuation_chunks=num_continuation_chunks, num_neighbours=num_neighbours, 
                                staleness_offset=staleness_offset, seq_len=seq_len, interval=interval))
                            has_unconsumed_retrieval = True
                end = time.time()
                time_per_step.append(end - start)

        else: # no staleness
            for i in range(total_steps):

                start = time.time()
                empty_inference(retrieved_tokens)
                
                if i > 0 and i % interval == 0:
                    
                    self.request = self.stub.Retrieve(retrieval_pb2.RetrievalRequest(
                        query=query, num_continuation_chunks=num_continuation_chunks, num_neighbours=num_neighbours, 
                        staleness_offset=staleness_offset, seq_len=seq_len, interval=interval))
                    response = await self.request
                    retrieved_tokens = response.retrieved_tokens
                    
                end = time.time()
                time_per_step.append(end - start)

            # print("RetrievalService client received: ", [str(r) for r in retrieved_tokens])
        print("Average Time per Step: {:.2f} ms".format(sum(time_per_step) / len(time_per_step) * 1000))
        print("Total Time: {:.2f} ms".format(sum(time_per_step) * 1000))
        # print(np.array(time_per_step) * 1000)
    
    # grpc client
    async def set_server_nprobe(self, nprobe):
        self.create_channel()
        nprobe_request = self.stub.SetNprobe(retrieval_pb2.SetNprobeRequest(nprobe=nprobe))
        nprobe_reply = await nprobe_request
        print("Server replied: " + nprobe_reply.reply)

if __name__ == '__main__':
    
    import argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--batch_size', type=int, default=1, help="batch size of the query")
    parser.add_argument('--staleness', type=int, default=1, help="whether to use staleness generation")

    parser.add_argument('--chunk_size', type=int, default=64, help="chunk size")
    parser.add_argument('--num_continuation_chunks', type=int, default=1, help="number of continuation chunks")
    parser.add_argument('--num_neighbours', type=int, default=1, help="number of neighbours")
    parser.add_argument('--staleness_offset', type=int, default=0, help="staleness offset")
    parser.add_argument('--seq_len', type=int, default=1, help="sequence length")
    parser.add_argument('--interval', type=int, default=64, help="interval")
    parser.add_argument('--nprobe', type=int, default=None, help="nprobe")

    args = parser.parse_args()

    retriever = ExternalRetriever(host=args.host, port=args.port, chunk_size=args.chunk_size)
    if args.nprobe is not None:
        asyncio.run(retriever.set_server_nprobe(args.nprobe))

    query = ['sample query'] * args.batch_size
    # for i in range(10):
    #     retrieved_tokens = asyncio.run(retriever.retrieve(query=query, 
    #         num_continuation_chunks=args.num_continuation_chunks, num_neighbours=args.num_neighbours, 
    #         staleness_offset=args.staleness_offset, seq_len=args.seq_len, interval=args.interval))
    #     print(retrieved_tokens)

    # Note! asyncio.run() will run everything, must wrap all the logic including channel and stub creating into a single 
    #  top-level function; cannot decouple channel creation, request send, and wait request in several components
    asyncio.run(retriever.dummy_generation(staleness=args.staleness, 
        query=query, num_continuation_chunks=args.num_continuation_chunks, num_neighbours=args.num_neighbours, 
        staleness_offset=args.staleness_offset, seq_len=args.seq_len, interval=args.interval))
