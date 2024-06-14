import asyncio
import logging

import grpc
import retrieval_pb2
import retrieval_pb2_grpc

import time
import numpy as np
from transformers import AutoTokenizer

# channel = grpc.aio.insecure_channel("localhost:50051")
# stub = retrieval_pb2_grpc.RetrievalServiceStub(channel)

localhost = '127.0.0.1'
# localhost = '172.31.40.69'
chunk_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)

async def run(nrun=10, **args) -> None:

    # async with grpc.aio.insecure_channel("localhost:50051") as channel:

    start = time.time()
    channel = grpc.aio.insecure_channel(f"{localhost}:50051") 
    stub = retrieval_pb2_grpc.RetrievalServiceStub(channel)
    end = time.time()
    print("Time to create channel: {:.2f} ms".format((end - start) * 1000))

    batch_size = len(args["query"])
    chunk_size = 64

    nprobe_request = stub.SetNprobe(retrieval_pb2.SetNprobeRequest(nprobe=32))
    nprobe_reply = await nprobe_request
    print("SetNprobe client received: " + nprobe_reply.reply)

    total_time = []
    wait_time = []
    for i in range(nrun):
        start = time.time()
        request = stub.Retrieve(retrieval_pb2.RetrievalRequest(query=args["query"], num_continuation_chunks=args["num_continuation_chunks"], 
            num_neighbours=args["num_neighbours"], staleness_offset=args["staleness_offset"], seq_len=args["seq_len"], interval=args["interval"], use_perf_model=args["use_perf_model"]))
        start_wait = time.time()
        print(f"Iter {i} Do something else after sending request")

        # the sync task must be executed in the following way to not block the progress
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, time.sleep, 0.1)

        response = await request
        end = time.time()
        print(len(response.retrieved_tokens))
        retrieved_tokens = np.array(response.retrieved_tokens, dtype=np.int32).reshape((batch_size, 1, args["num_neighbours"], (1 + args["num_continuation_chunks"]) * chunk_size))

        print("RetrievalService client received: ")
        for i, query in enumerate(args["query"]):
            print(f"\nQuery {i}: {query}")
            print("\nRetrieved tokens:")
            for n in range(args["num_neighbours"]):
                print("\nNeighbour {}: {}".format(n, chunk_tokenizer.decode(retrieved_tokens[i][0][n])))
        print("Total Time: {:.2f} ms".format((end - start) * 1000))
        print("Wait Time: {:.2f} ms".format((end - start_wait) * 1000))
        total_time.append(end - start)
        wait_time.append(end - start_wait)
    
    print("Average Total Time: {:.2f} ms".format(sum(total_time) / len(total_time) * 1000))
    print("Average Wait Time: {:.2f} ms".format(sum(wait_time) / len(wait_time) * 1000))

if __name__ == "__main__":

    logging.basicConfig()
    queries = ["The rich get richer and the poor get poorer eh?\nOr is it the rich think different and play by a different set of rules?\nDo the rich take responsibility and action?\nPoor people believe 'Life happens to me.' Rich people are committed to be rich.\nPoor people WANT to be rich. Rich people think big.\nPoor people think small. Rich people focus on opportunities.\nPoor people focus on obstacles. Rich people are willing to promote themselves and their value.\nPoor people think negatively about selling and promotion.\nPoor people are closed to new ideas..\nDo You think rich or poor?",
        "The Denver Board of Education opened the 2017-18 school year with an update on projects that include new construction, upgrades, heat mitigation and quality learning environments.\nWe are excited that Denver students will be the beneficiaries of a four year, $572 million General Obligation Bond. Since the passage of the bond, our construction team has worked to schedule the projects over the four-year term of the bond.\nDenver voters on Tuesday approved bond and mill funding measures for students in Denver Public Schools, agreeing to invest $572 million in bond funding to build and improve schools and $56.6 million in operating dollars to support proven initiatives, such as early literacy.\nDenver voters say yes to bond and mill levy funding support for DPS students and schools."]
    asyncio.run(run(nrun=10, query=queries, num_continuation_chunks=1, num_neighbours=2, staleness_offset=0, seq_len=128, interval=64, use_perf_model=True))