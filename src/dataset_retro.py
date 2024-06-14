import numpy as np
import torch
from typing import List
from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset
import sys

sys.path.append('../inference')
from performance_model import PerformanceModel

class ChunkedSequenceDataset:

    def __init__(self, chunks: Path, seq2chunk: Path, chunk2seq: Path):
        self.chunks_path = chunks
        self.chunks = np.load(str(chunks), mmap_mode="r")
        self.seq2chunk = np.load(str(seq2chunk), mmap_mode="r")
        self.chunk2seq = np.load(str(chunk2seq), mmap_mode="r")

    @property
    def chunk_size(self):
        return self.chunks.shape[1]

    @property
    def num_chunks(self):
        return self.chunks.shape[0]

    @property
    def num_sequences(self):
        return self.seq2chunk.shape[0]

    def get_chunk_indices_of_sequence(self, sequence_index):
        chunk_start_idx = self.seq2chunk[sequence_index]
        if sequence_index + 1 < self.seq2chunk.shape[0]:
            chunk_end_idx = self.seq2chunk[sequence_index + 1]
        else:
            # this is the last sequence in the shard
            chunk_end_idx = self.chunks.shape[0]
        return np.arange(chunk_start_idx, chunk_end_idx)

    def get_chunk_tokens(self, chunk_index, include_continuation_chunks=0):
        start_idx = chunk_index
        end_idx = chunk_index + 1
        while end_idx - start_idx - 1 < include_continuation_chunks and \
            end_idx < len(self.chunk2seq) and \
            self.chunk2seq[start_idx] == self.chunk2seq[end_idx]:
            end_idx += 1
        return self.chunks[start_idx:end_idx, :].reshape(-1)


class ShardedChunkedSequenceDataset:

    def __init__(self, shards: List[ChunkedSequenceDataset]):
        self.shards = shards
        assert all(shard.chunk_size == shards[0].chunk_size for shard in shards), \
            "All shards must have same chunk size"

        self.shard_seq_ranges = []
        self.shard_chunk_ranges = []
        self.total_num_chunks = 0
        self.total_num_sequences = 0
        for shard in shards:
            self.shard_seq_ranges.append(range(self.total_num_sequences, self.total_num_sequences + shard.num_sequences))
            self.shard_chunk_ranges.append(range(self.total_num_chunks, self.total_num_chunks + shard.num_chunks))
            self.total_num_sequences += shard.num_sequences
            self.total_num_chunks += shard.num_chunks

    @property
    def chunk_size(self):
        return self.shards[0].chunk_size

    @property
    def num_chunks(self):
        return self.total_num_chunks

    @property
    def num_sequences(self):
        return self.total_num_sequences

    def get_chunk_indices_of_sequence(self, sequence_index):
        for shard_seq_range, shard_chunk_range, shard in zip(self.shard_seq_ranges, self.shard_chunk_ranges, self.shards):
            if int(sequence_index) in shard_seq_range:
                local_seq_index = sequence_index - shard_seq_range.start
                return shard_chunk_range.start + shard.get_chunk_indices_of_sequence(local_seq_index)
        raise IndexError(f"Sequence with index {sequence_index} not found in index")

    def get_chunk_tokens(self, chunk_index, include_continuation_chunks: int=0):
        for shard_range, shard in zip(self.shard_chunk_ranges, self.shards):
            if int(chunk_index) in shard_range:
                local_chunk_index = chunk_index - shard_range.start
                return shard.get_chunk_tokens(local_chunk_index, include_continuation_chunks)
        raise IndexError(f"Chunk with index {chunk_index} not found in index")

    def get_sequence_index_of_chunk(self, chunk_index):
        for shard_range, shard in zip(self.shard_chunk_ranges, self.shards):
            if int(chunk_index) in shard_range:
                local_chunk_index = chunk_index - shard_range.start
                return shard.chunk2seq[local_chunk_index]
        raise IndexError(f"Chunk with index {chunk_index} not found in index")

class ChunkNeighbourDataset:

    def __init__(self, neighbours: Path, retrieval_dataset: ShardedChunkedSequenceDataset):
        self.neighbours = np.load(str(neighbours), mmap_mode="r")
        self.retrieval_dataset = retrieval_dataset

    @property
    def chunk_size(self):
        return self.retrieval_dataset.chunk_size

    def __len__(self):
        return self.neighbours.shape[0]

    def get_neighbours(self, chunk_index: int, num_neighbours: int=None, continuation_chunks: int=1):
        """
        Returns precomputed tokens for all neighbours of chunk.
        Shape: [num_neighbours, chunk_size * (1 + continuation_chunks)]
        """
        return [
            self.retrieval_dataset.get_chunk_tokens(
                neighbour_chunk_idx, 
                include_continuation_chunks=continuation_chunks
            ) if neighbour_chunk_idx != -1 else None
            for neighbour_chunk_idx in self.neighbours[chunk_index][:num_neighbours]
        ]
            

class ShardedChunkNeighbourDataset:

    def __init__(self, shards: List[ChunkNeighbourDataset]):
        self.shards = shards
        assert all(shard.chunk_size == shards[0].chunk_size for shard in shards), \
            "The chunk size in all shards must match"

        self.shard_ranges = []
        self.total = 0
        for shard in shards:
            self.shard_ranges.append(range(self.total, self.total + len(shard)))
            self.total += len(shard)

    @property
    def chunk_size(self):
        return self.shards[0].chunk_size

    def __len__(self):
        return self.total

    def get_neighbours(self, chunk_index: int, num_neighbours: int=None, continuation_chunks: int=1):
        for shard_range, shard in zip(self.shard_ranges, self.shards):
            if int(chunk_index) in shard_range:
                local_index = chunk_index - shard_range.start
                return shard.get_neighbours(local_index, num_neighbours, continuation_chunks)
        raise IndexError(f"Neighbours for index {chunk_index} not found")


RetroTrainingExample = namedtuple("RetroTrainingExample", [
    "input_ids", 
    "neighbour_ids", 
    "labels"
])

class RetroDataset(Dataset):

    def __init__(
        self, 
        input_dataset: ShardedChunkedSequenceDataset, 
        neighbour_dataset: ShardedChunkNeighbourDataset, 
        num_neighbours=None, 
        continuation_chunks=1, 
        pad_token_idx=0,
        max_len=None
    ):
        super().__init__()
        self.input_dataset = input_dataset
        self.neighbour_dataset = neighbour_dataset
        self.num_neighbours = num_neighbours
        self.continuation_chunks = continuation_chunks
        self.neighbour_size = neighbour_dataset.chunk_size * (1 + continuation_chunks)
        self.pad_token_idx = pad_token_idx
        self.max_num_chunks = max_len // input_dataset.chunk_size if max_len is not None else None

        if max_len is not None:
            assert max_len % input_dataset.chunk_size == 0, \
                "max_len must be a multiple of chunk_size"

        assert input_dataset.num_chunks == len(neighbour_dataset), \
            "The number of chunks in input dataset did not match the number of chunks in neighbour dataset"

    def __len__(self):
        return self.input_dataset.num_sequences

    def __getitem__(self, seq_index: int) -> RetroTrainingExample:
        input_chunk_indices = self.input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids
        input_ids = np.concatenate([
            self.input_dataset.get_chunk_tokens(chunk_index)
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # neighbour_ids
        neighbour_ids = np.stack([
            [
                np.pad(neighbour_tokens, (0, self.neighbour_size - len(neighbour_tokens)), constant_values=self.pad_token_idx) \
                    if neighbour_tokens is not None else \
                np.ones(self.neighbour_size) * self.pad_token_idx

                for neighbour_tokens in self.neighbour_dataset.get_neighbours(
                    chunk_index, 
                    num_neighbours=self.num_neighbours, 
                    continuation_chunks=self.continuation_chunks
                )
            ]
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # labels - set to -100 at padded tokens
        labels = np.pad(input_ids[1:], (0, 1), constant_values=self.pad_token_idx).astype(np.int64)
        labels[labels == self.pad_token_idx] = -100

        return RetroTrainingExample(
            torch.from_numpy(input_ids.astype(np.int32)), 
            torch.from_numpy(neighbour_ids.astype(np.int32)), 
            torch.from_numpy(labels)
        )


class RetroDatasetRetrieveRealTime(Dataset):

    def __init__(
        self, 
        input_dataset: ShardedChunkedSequenceDataset, 
        retrieval_dataset: ShardedChunkedSequenceDataset, 
        index, # Faiss index
        query_chunk_embeddings, # a list of input_dataset.num_sequences elements, each is a numpy array of shape (num_query_chunks_of_seq, dim)
        num_neighbours=None, 
        continuation_chunks=1, 
        pad_token_idx=0,
        max_len=None,
        no_retrieval=False,
        retrieval_interval=None
    ):
        super().__init__()
        self.input_dataset = input_dataset
        self.retrieval_dataset = retrieval_dataset
        self.chunk_size = input_dataset.chunk_size

        # the index objects should be loaded and optionally moved to GPU before passing to this class
        assert index.is_trained, "The index must be trained"
        self.index = index

        self.query_chunk_embeddings = query_chunk_embeddings

        # assume neighbor chunk size = input chunk size
        self.num_neighbours = num_neighbours
        self.continuation_chunks = continuation_chunks
        self.neighbour_size = input_dataset.chunk_size * (1 + continuation_chunks)
        self.pad_token_idx = pad_token_idx
        self.max_num_chunks = max_len // input_dataset.chunk_size if max_len is not None else None
        self.no_retrieval = no_retrieval
        self.retrieval_interval = self.chunk_size if retrieval_interval is None else retrieval_interval

        if max_len is not None:
            assert max_len % input_dataset.chunk_size == 0, \
                "max_len must be a multiple of chunk_size"
            
    def __len__(self):
        return self.input_dataset.num_sequences

    def __getitem__(self, seq_index: int) -> RetroTrainingExample:
        input_chunk_indices = self.input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids, if the sequence length <= max_num_chunks, keep all, otherwise trunk to max_num_chunks
        input_ids = np.concatenate([
            self.input_dataset.get_chunk_tokens(chunk_index)
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # search for neighbors
        query_embeddings = self.query_chunk_embeddings[seq_index]
        max_num_query_chunks = int(np.ceil((input_ids.shape[0] - self.chunk_size) / self.retrieval_interval)) + 1
        num_query_chunks = query_embeddings.shape[0] if query_embeddings.shape[0] < max_num_query_chunks else max_num_query_chunks
        query_embeddings = query_embeddings[:num_query_chunks]

        _, retrieved_chunk_indices = self.index.search(query_embeddings, self.num_neighbours)
        # print("num_query_chunks", num_query_chunks)
        # print("retrieved_chunk_indices", retrieved_chunk_indices)

        # stack neighbor ids
        neighbour_ids = []
        for i in range(num_query_chunks):
            neighbour_ids_per_query = []
            for neighbour_chunk_idx in retrieved_chunk_indices[i]:
                neighbour_tokens = self.retrieval_dataset.get_chunk_tokens(
                    neighbour_chunk_idx, 
                    include_continuation_chunks=self.continuation_chunks
                )
                neighbour_ids_per_query.append(np.pad(neighbour_tokens, (0, self.neighbour_size - len(neighbour_tokens)), constant_values=self.pad_token_idx) \
                    if (neighbour_tokens is not None and not self.no_retrieval) else \
                np.ones(self.neighbour_size) * self.pad_token_idx)
                
            neighbour_ids.append(neighbour_ids_per_query)
        neighbour_ids = np.stack(neighbour_ids)

        # labels - set to -100 at padded tokens
        labels = np.pad(input_ids[1:], (0, 1), constant_values=self.pad_token_idx).astype(np.int64)
        labels[labels == self.pad_token_idx] = -100

        return RetroTrainingExample(
            torch.from_numpy(input_ids.astype(np.int32)), 
            torch.from_numpy(neighbour_ids.astype(np.int32)), 
            torch.from_numpy(labels)
        )


class RetroDatasetRetrieveRealTimeStale(RetroDatasetRetrieveRealTime):
    """ Allow staleness during retrieval """

    def __init__(
        self, 
        input_dataset: ShardedChunkedSequenceDataset, 
        retrieval_dataset: ShardedChunkedSequenceDataset, 
        index, # Faiss index
        query_chunk_embeddings, # a list of input_dataset.num_sequences elements, each is a numpy array of shape (num_query_chunks_of_seq, dim)
        num_neighbours=None, 
        continuation_chunks=1, 
        pad_token_idx=0,
        max_len=None,
        remove_stale_context=False,
        no_retrieval=False,
        retrieval_interval=None
    ):
        super().__init__(
            input_dataset=input_dataset,
            retrieval_dataset=retrieval_dataset,
            index=index,
            query_chunk_embeddings=query_chunk_embeddings,
            num_neighbours=num_neighbours,
            continuation_chunks=continuation_chunks,
            pad_token_idx=pad_token_idx,
            max_len=max_len,
            no_retrieval=no_retrieval,
            retrieval_interval=retrieval_interval
        )
        self.remove_stale_context = remove_stale_context
    
    def __getitem__(self, seq_index: int) -> RetroTrainingExample:
        input_chunk_indices = self.input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids, if the sequence length <= max_num_chunks, keep all, otherwise trunk to max_num_chunks
        input_ids = np.concatenate([
            self.input_dataset.get_chunk_tokens(chunk_index)
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # search for neighbors
        query_embeddings = self.query_chunk_embeddings[seq_index]
        max_num_query_chunks = int(np.ceil((input_ids.shape[0] - self.chunk_size) / self.retrieval_interval)) + 1
        num_query_chunks = query_embeddings.shape[0] if query_embeddings.shape[0] < max_num_query_chunks else max_num_query_chunks
        query_embeddings = query_embeddings[:num_query_chunks]
        for i in reversed(range(1, num_query_chunks)): # starting from the second query, use stale context
            query_embeddings[i] = query_embeddings[i - 1]

        # original_chunk_indices = input_chunk_indices[:self.max_num_chunks]
        # # the first chunk of the sequence is not stale, while the rest are stale
        # stale_chunk_indices = [idx - 1 if i > 0 else idx for i, idx in enumerate(original_chunk_indices)] 
        # query_embeddings = self.query_chunk_embeddings[stale_chunk_indices]

        _, retrieved_chunk_indices = self.index.search(query_embeddings, self.num_neighbours)

        # stack neighbor ids
        neighbour_ids = []

        """
        Behavior of staleness:
            continuation_chunks == total chunks - 1, setting continuation_chunks same as the non-staleness version will ensure this.
            remove_stale_context -> if true, shifting the starting point retrieved content, and still get (continuation_chunks + 1) * chunk_size of tokens. 
                The shifting length is related to the staleness, e.g., retrieval interval = 64, staleness = 64, shift 64; retrieval interval = 16, staleness = 16, shift 16.
        """
        for i in range(num_query_chunks):

            neighbour_ids_per_query = []
            for neighbour_chunk_idx in retrieved_chunk_indices[i]:
                neighbour_tokens = self.retrieval_dataset.get_chunk_tokens(
                    neighbour_chunk_idx, # the continuous start from old 
                    include_continuation_chunks=self.continuation_chunks + 1 # retrieve an extra chunk for trimming
                )
                if (neighbour_tokens is not None and not self.no_retrieval):
                    retrieved_tokens = np.pad(neighbour_tokens, (0, self.neighbour_size + self.chunk_size - len(neighbour_tokens)), constant_values=self.pad_token_idx)
                else:
                    retrieved_tokens = np.ones(self.neighbour_size + self.chunk_size) * self.pad_token_idx
                if self.remove_stale_context and i > 0: # remove the context of the stale query
                # if self.remove_stale_context: # remove the context of the stale query
                    if self.chunk_size != self.retrieval_interval:
                        retrieved_tokens = retrieved_tokens[self.retrieval_interval: -(self.chunk_size - self.retrieval_interval)] 
                    else:
                        retrieved_tokens = retrieved_tokens[self.chunk_size:]
                else: # keep the original retrieved content, except the extra retrieved chunk
                    retrieved_tokens = retrieved_tokens[:-self.input_dataset.chunk_size]
                neighbour_ids_per_query.append(retrieved_tokens)
                
            neighbour_ids.append(neighbour_ids_per_query)
        neighbour_ids = np.stack(neighbour_ids)


        # for i in range(num_query_chunks):
        #     neighbour_ids_per_query = []
        #     for neighbour_chunk_idx in retrieved_chunk_indices[i]:
        #         neighbour_tokens = self.retrieval_dataset.get_chunk_tokens(
        #             neighbour_chunk_idx, # the continuous start from old 
        #             include_continuation_chunks=self.continuation_chunks
        #         )
        #         if self.remove_stale_context: # remove the context of the stale query
        #             neighbour_tokens = neighbour_tokens[self.input_dataset.chunk_size:] 
        #         neighbour_ids_per_query.append(np.pad(neighbour_tokens, (0, self.neighbour_size - len(neighbour_tokens)), constant_values=self.pad_token_idx) \
        #             if (neighbour_tokens is not None and not self.no_retrieval) else \
        #         np.ones(self.neighbour_size) * self.pad_token_idx)
                
        #     neighbour_ids.append(neighbour_ids_per_query)
        # neighbour_ids = np.stack(neighbour_ids)

        # labels - set to -100 at padded tokens
        labels = np.pad(input_ids[1:], (0, 1), constant_values=self.pad_token_idx).astype(np.int64)
        labels[labels == self.pad_token_idx] = -100

        return RetroTrainingExample(
            torch.from_numpy(input_ids.astype(np.int32)), 
            torch.from_numpy(neighbour_ids.astype(np.int32)), 
            torch.from_numpy(labels)
        )


class RetroDatasetRetrieveRealTimeStaleUsePerfModel(RetroDatasetRetrieveRealTime):
    """ Allow staleness during retrieval """

    def __init__(
        self, 
        input_dataset: ShardedChunkedSequenceDataset, 
        retrieval_dataset: ShardedChunkedSequenceDataset, 
        index, # Faiss index
        query_chunk_embeddings, # a list of input_dataset.num_sequences elements, each is a numpy array of shape (num_query_chunks_of_seq, dim)
        num_neighbours=None, 
        continuation_chunks=1, 
        pad_token_idx=0,
        max_len=None,
        remove_stale_context=False,
        no_retrieval=False,
        retrieval_interval=None,
        # performance model
        generation_model_path : Path = Path('$WORKSPACE/inference/performance/performance_generation_len_1024_k_2.pickle'),
        retrieval_model_path : Path = Path('$WORKSPACE/inference/performance/performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'),
        sbert_model_path : Path = Path('$WORKSPACE/inference/performance/performance_SBERT.pickle'),
        extra_overhead_ms = 10, # set a default extra latency on the retrieval side
        search_latency_budget_discount = 1.0, # if < 1.0, e.g., 0.9, limit the latency budget of search to 90%
        min_nprobe = None,
        max_nprobe = None
    ):
        super().__init__(
            input_dataset=input_dataset,
            retrieval_dataset=retrieval_dataset,
            index=index,
            query_chunk_embeddings=query_chunk_embeddings,
            num_neighbours=num_neighbours,
            continuation_chunks=continuation_chunks,
            pad_token_idx=pad_token_idx,
            max_len=max_len,
            no_retrieval=no_retrieval,
            retrieval_interval=retrieval_interval
        )
        self.index.parallel_mode = 1 # intra-query parallelism
        self.remove_stale_context = remove_stale_context
        self.perf_model = PerformanceModel(
            generation_model_path=generation_model_path, retrieval_model_path=retrieval_model_path, sbert_model_path=sbert_model_path,
            extra_overhead_ms=extra_overhead_ms, search_latency_budget_discount=search_latency_budget_discount,
            min_nprobe=min_nprobe, max_nprobe=max_nprobe)
    
    
    def __getitem__(self, seq_index: int) -> RetroTrainingExample:
        input_chunk_indices = self.input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids, if the sequence length <= max_num_chunks, keep all, otherwise trunk to max_num_chunks
        input_ids = np.concatenate([
            self.input_dataset.get_chunk_tokens(chunk_index)
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # search for neighbors
        query_embeddings = self.query_chunk_embeddings[seq_index]
        max_num_query_chunks = int(np.ceil((input_ids.shape[0] - self.chunk_size) / self.retrieval_interval)) + 1
        num_query_chunks = query_embeddings.shape[0] if query_embeddings.shape[0] < max_num_query_chunks else max_num_query_chunks
        query_embeddings = query_embeddings[:num_query_chunks]
        for i in reversed(range(1, num_query_chunks)): # starting from the second query, use stale context
            query_embeddings[i] = query_embeddings[i - 1]

        # search queries one-by-one, use the performance model to predict the nprobe
        retrieved_chunk_indices = []
        for i in range(num_query_chunks):
            seq_len = self.chunk_size + i * self.retrieval_interval
            nprobe = self.perf_model.predict(seq_len, self.retrieval_interval)
            self.index.nprobe = nprobe
            _, retrieved_chunk_indices_per_query = self.index.search(query_embeddings[i].reshape(1, -1), self.num_neighbours)
            retrieved_chunk_indices.append(retrieved_chunk_indices_per_query[0])

        retrieved_chunk_indices = np.stack(retrieved_chunk_indices)
        assert retrieved_chunk_indices.shape == (num_query_chunks, self.num_neighbours)
        # _, retrieved_chunk_indices = self.index.search(query_embeddings, self.num_neighbours)

        # stack neighbor ids
        neighbour_ids = []

        """
        Behavior of staleness:
            continuation_chunks == total chunks - 1, setting continuation_chunks same as the non-staleness version will ensure this.
            remove_stale_context -> if true, shifting the starting point retrieved content, and still get (continuation_chunks + 1) * chunk_size of tokens. 
                The shifting length is related to the staleness, e.g., retrieval interval = 64, staleness = 64, shift 64; retrieval interval = 16, staleness = 16, shift 16.
        """
        for i in range(num_query_chunks):

            neighbour_ids_per_query = []
            for neighbour_chunk_idx in retrieved_chunk_indices[i]:
                neighbour_tokens = self.retrieval_dataset.get_chunk_tokens(
                    neighbour_chunk_idx, # the continuous start from old 
                    include_continuation_chunks=self.continuation_chunks + 1 # retrieve an extra chunk for trimming
                )
                if (neighbour_tokens is not None and not self.no_retrieval):
                    retrieved_tokens = np.pad(neighbour_tokens, (0, self.neighbour_size + self.chunk_size - len(neighbour_tokens)), constant_values=self.pad_token_idx)
                else:
                    retrieved_tokens = np.ones(self.neighbour_size + self.chunk_size) * self.pad_token_idx
                if self.remove_stale_context and i > 0: # remove the context of the stale query
                # if self.remove_stale_context: # remove the context of the stale query
                    if self.chunk_size != self.retrieval_interval:
                        retrieved_tokens = retrieved_tokens[self.retrieval_interval: -(self.chunk_size - self.retrieval_interval)] 
                    else:
                        retrieved_tokens = retrieved_tokens[self.chunk_size:]
                else: # keep the original retrieved content, except the extra retrieved chunk
                    retrieved_tokens = retrieved_tokens[:-self.input_dataset.chunk_size]
                neighbour_ids_per_query.append(retrieved_tokens)
                
            neighbour_ids.append(neighbour_ids_per_query)
        neighbour_ids = np.stack(neighbour_ids)

        labels = np.pad(input_ids[1:], (0, 1), constant_values=self.pad_token_idx).astype(np.int64)
        labels[labels == self.pad_token_idx] = -100

        return RetroTrainingExample(
            torch.from_numpy(input_ids.astype(np.int32)), 
            torch.from_numpy(neighbour_ids.astype(np.int32)), 
            torch.from_numpy(labels)
        )


class RetroDatasetRetrieveRealTimeRetrieveOnce(RetroDatasetRetrieveRealTime):
    """ RETRO, but only retrieve once at the beginning """

    def __init__(
        self, 
        input_dataset: ShardedChunkedSequenceDataset, 
        retrieval_dataset: ShardedChunkedSequenceDataset, 
        index, # Faiss index
        query_chunk_embeddings, # a list of input_dataset.num_sequences elements, each is a numpy array of shape (num_query_chunks_of_seq, dim)
        num_neighbours=None, 
        continuation_chunks=1, 
        pad_token_idx=0,
        max_len=None,
        no_retrieval=False,
        retrieval_interval=None
    ):
        super().__init__(
            input_dataset=input_dataset,
            retrieval_dataset=retrieval_dataset,
            index=index,
            query_chunk_embeddings=query_chunk_embeddings,
            num_neighbours=num_neighbours,
            continuation_chunks=continuation_chunks,
            pad_token_idx=pad_token_idx,
            max_len=max_len,
            no_retrieval=no_retrieval,
            retrieval_interval=retrieval_interval
        )
    
    def __getitem__(self, seq_index: int) -> RetroTrainingExample:
        input_chunk_indices = self.input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids, if the sequence length <= max_num_chunks, keep all, otherwise trunk to max_num_chunks
        input_ids = np.concatenate([
            self.input_dataset.get_chunk_tokens(chunk_index)
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # search for neighbors
        query_embeddings = self.query_chunk_embeddings[seq_index]
        max_num_query_chunks = int(np.ceil((input_ids.shape[0] - self.chunk_size) / self.retrieval_interval)) + 1
        num_query_chunks = query_embeddings.shape[0] if query_embeddings.shape[0] < max_num_query_chunks else max_num_query_chunks
        for i in reversed(range(1, num_query_chunks)): # starting from the second query, use stale context
            query_embeddings[i] = query_embeddings[0]

        # original_chunk_indices = input_chunk_indices[:self.max_num_chunks]
        # # the first chunk of the sequence is not stale, while the rest are stale
        # stale_chunk_indices = [idx - 1 if i > 0 else idx for i, idx in enumerate(original_chunk_indices)] 
        # query_embeddings = self.query_chunk_embeddings[stale_chunk_indices]

        _, retrieved_chunk_indices = self.index.search(query_embeddings, self.num_neighbours)

        # stack neighbor ids
        neighbour_ids = []
        for i in range(num_query_chunks):
            neighbour_ids_per_query = []
            for neighbour_chunk_idx in retrieved_chunk_indices[i]:
                neighbour_tokens = self.retrieval_dataset.get_chunk_tokens(
                    neighbour_chunk_idx, 
                    include_continuation_chunks=self.continuation_chunks
                )
                neighbour_ids_per_query.append(np.pad(neighbour_tokens, (0, self.neighbour_size - len(neighbour_tokens)), constant_values=self.pad_token_idx) \
                    if (neighbour_tokens is not None and not self.no_retrieval) else \
                np.ones(self.neighbour_size) * self.pad_token_idx)
                
            neighbour_ids.append(neighbour_ids_per_query)
        neighbour_ids = np.stack(neighbour_ids)

        # labels - set to -100 at padded tokens
        labels = np.pad(input_ids[1:], (0, 1), constant_values=self.pad_token_idx).astype(np.int64)
        labels[labels == self.pad_token_idx] = -100

        return RetroTrainingExample(
            torch.from_numpy(input_ids.astype(np.int32)), 
            torch.from_numpy(neighbour_ids.astype(np.int32)), 
            torch.from_numpy(labels)
        )
