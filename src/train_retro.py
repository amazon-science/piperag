import argparse
import torch
import torch.nn.functional as F
import json
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from typing import List, Optional
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from modeling_retro import RetroModelLMHead, RetroConfig
from modeling_retro_inference import RetroModelLMHeadInference
from dataset_retro import RetroDataset, ChunkedSequenceDataset, RetroTrainingExample, ShardedChunkedSequenceDataset, \
    ChunkNeighbourDataset, ShardedChunkNeighbourDataset, \
    RetroDatasetRetrieveRealTime, RetroDatasetRetrieveRealTimeStale, \
    RetroDatasetRetrieveRealTimeStaleUsePerfModel, RetroDatasetRetrieveRealTimeRetrieveOnce
from retrieval import Retriever

import faiss
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
import gc

def get_realtime_retrieval_retro_dataset_from_spec(
    spec_file: Path, 
    num_neighbours=None,
    continuation_chunks=1,
    pad_token_idx=0,
    max_len=None,
    use_gpus=False,
    staleness=False,
    stale_steps=None,
    remove_stale_context=False,
    no_retrieval=False, # no retrieval during generatoin
    one_retrieval=False, # only one retrieval at the beginning of the generation
    retrieval_interval=64,
    nprobe=None,

    use_perf_model=False,
    # the following arguments will only be used when use_perf_model=True
    generation_model_path : Path = Path('$WORKSPACE/inference/performance/performance_generation_len_1024_k_2.pickle'),
    retrieval_model_path : Path = Path('$WORKSPACE/inference/performance/performance_search_c4_chunk_0_to_999_IVF16384,PQ64.pickle'),
    sbert_model_path : Path = Path('$WORKSPACE/inference/performance/performance_SBERT.pickle'),
    extra_overhead_ms = 10, # set a default extra latency on the retrieval side
    search_latency_budget_discount = 1.0, # if < 1.0, e.g., 0.9, limit the latency budget of search to 90%
    min_nprobe = None,
    max_nprobe = None

) -> RetroDatasetRetrieveRealTime:

    spec = json.load(spec_file.open())
    base_dir = spec_file.parent
    if staleness and stale_steps is None:
        stale_steps = retrieval_interval
        print("stale_steps is None, set it as retrieval_interval {}".format(stale_steps))
    print("staleness: {}, stale_steps: {}".format(staleness, stale_steps))

    if staleness and stale_steps != retrieval_interval: # a special case: keep the same interval, only do staleness shift
        raise NotImplementedError("the correct version of staleness shift not implemented yet (no according attention mechanism support)")

    print("Load index...")
    start = time.time()
    index = faiss.read_index(str(base_dir / spec["neighbours"]["faiss_index"]))
    assert index.is_trained, "The index must be trained"
    if use_gpus:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = args.shard_index
        index = faiss.index_cpu_to_all_gpus(index, co) 
    # else:
    #     index.parallel_mode = 1 # intra-query parallelism
    # priority: input arg > spec 
    print("Load index takes {} seconds".format(time.time() - start))

    if nprobe is not None:
        index.nprobe = nprobe
        print("Find nprobe in input args, override the one in spec")
        print("Set nprobe as {}".format(index.nprobe))
    elif "nprobe" in spec["neighbours"]: 
        index.nprobe = int(spec["neighbours"]["nprobe"])
        print("No nprobe in input args, find it in spec")
        print("Set nprobe as {}".format(index.nprobe))
    else:
        print("No nprobe in input args or spec, use default nprobe {}".format(index.nprobe))

    # input dataset
    input_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=base_dir / shard["chunks"],
            seq2chunk=base_dir / shard["seq2chunk"],
            chunk2seq=base_dir / shard["chunk2seq"]
        )
        for shard in spec["shards"]
    ])

    chunk_size = input_dataset.chunk_size
    

    # # encode all the query chunk embeddings here
    print("Loading Sentence BERT...")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        device = "cuda"
        # device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        model_st.to(device)

    print("Generating sentence embeddings...")
    chunk_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)
    query_chunk_embeddings = [] # a list of input_dataset.num_sequences elements, each is a numpy array of shape (num_query_chunks_of_seq, dim)
    for seq_index in range(input_dataset.num_sequences):
        # print("Sequence index: ", seq_index)
        input_chunk_indices = input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids, if the sequence length <= max_num_chunks, keep all, otherwise trunk to max_num_chunks
        input_ids = [input_dataset.get_chunk_tokens(chunk_index) for chunk_index in input_chunk_indices]
        input_ids = np.concatenate(input_ids, axis=0)
        num_query_chunks = int(np.ceil((input_ids.shape[0] - chunk_size) / retrieval_interval)) + 1
        query_window_tokens = []
        if staleness and stale_steps != retrieval_interval: # a special case: keep the same interval, only do staleness shift
            raise NotImplementedError("the correct version of staleness shift not implemented yet (no according attention mechanism support)")
            # do the staleness processing at the query level, later on run the normal, non-stale evaluation
            assert stale_steps <= retrieval_interval
            query_window_tokens.append(chunk_tokenizer.decode(input_ids[: chunk_size], skip_special_tokens=True))
            for i in range(1, num_query_chunks):
                query_window_tokens.append(chunk_tokenizer.decode(input_ids[i * retrieval_interval - stale_steps: i * retrieval_interval - stale_steps + chunk_size], skip_special_tokens=True))
            query_chunk_embeddings.append(model_st.encode(query_window_tokens, convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True))
        else: # no staleness or stale_steps == retrieval interval (staleness handled in RetroDatasetRetrieveRealTimeStale)
            for i in range(num_query_chunks):
                query_window_tokens.append(chunk_tokenizer.decode(input_ids[i * retrieval_interval: i * retrieval_interval + chunk_size], skip_special_tokens=True))
            query_chunk_embeddings.append(model_st.encode(query_window_tokens, convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True))

    # # print("Load chunk embeddings...")
    # query_chunk_embeddings_list = [np.load(base_dir /shard["embeddings"]).astype("float32") for shard in spec["shards"]]
    # query_chunk_embeddings_loaded = np.concatenate(query_chunk_embeddings_list, axis=0)
    # no, they are not the same, because they have different orders
    # assert np.allclose(query_chunk_embeddings_loaded, np.concatenate(query_chunk_embeddings, axis=0)), "Loaded chunk embeddings are not the same as the generated ones"

    # delete the model to avoid fork error in pytorch lignthing
    model_st.to("cpu")
    del model_st
    gc.collect()
    torch.cuda.empty_cache()

        
    
    # query_window_tokens = np.array(query_window_tokens)
    # query_embeddings = self.model_st.encode(query_window_tokens, convert_to_numpy=True, output_value="sentence_embedding", normalize_embeddings=True)

    # retrieval dataset
    index_spec = json.load((base_dir / spec["neighbours"]["index_spec"]).open())
    index_base_dir = base_dir / Path(spec["neighbours"]["index_spec"]).parent
    retrieval_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=index_base_dir / shard["chunks"],
            seq2chunk=index_base_dir / shard["seq2chunk"],
            chunk2seq=index_base_dir / shard["chunk2seq"]
        )
        for shard in index_spec
    ])

    if one_retrieval:
        print("Only one retrieval at the beginning of the generation")
        retro_dataset = RetroDatasetRetrieveRealTimeRetrieveOnce(
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
    elif not staleness:
        print("No staleness")
        retro_dataset = RetroDatasetRetrieveRealTime(
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
    else:
        if not use_perf_model:
            if stale_steps == retrieval_interval:
                print("With staleness, using fixed nprobe")
                retro_dataset = RetroDatasetRetrieveRealTimeStale(
                    input_dataset=input_dataset,
                    retrieval_dataset=retrieval_dataset,
                    index=index, 
                    query_chunk_embeddings=query_chunk_embeddings,
                    num_neighbours=num_neighbours,
                    continuation_chunks=continuation_chunks,
                    pad_token_idx=pad_token_idx,
                    max_len=max_len,
                    remove_stale_context=remove_stale_context,
                    no_retrieval=no_retrieval,
                    retrieval_interval=retrieval_interval
                )
            else:
                print("With staleness, using fixed nprobe; stale steps {} != retrieval interval {}".format(stale_steps, retrieval_interval))
                # do the staleness processing at the query level, later on run the normal, non-stale evaluation
                retro_dataset = RetroDatasetRetrieveRealTime(
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
        else:
            print("With staleness, using perf model")
            retro_dataset = RetroDatasetRetrieveRealTimeStaleUsePerfModel(
                input_dataset=input_dataset,
                retrieval_dataset=retrieval_dataset,
                index=index, 
                query_chunk_embeddings=query_chunk_embeddings,
                num_neighbours=num_neighbours,
                continuation_chunks=continuation_chunks,
                pad_token_idx=pad_token_idx,
                max_len=max_len,
                remove_stale_context=remove_stale_context,
                no_retrieval=no_retrieval,
                retrieval_interval=retrieval_interval,
                # performance model
                generation_model_path=generation_model_path,
                retrieval_model_path=retrieval_model_path,
                sbert_model_path=sbert_model_path,
                extra_overhead_ms=extra_overhead_ms,
                search_latency_budget_discount=search_latency_budget_discount,
                min_nprobe=min_nprobe,
                max_nprobe=max_nprobe
            )
        
    return retro_dataset

def get_retro_dataset_from_spec(
    spec_file: Path, 
    num_neighbours=None,
    continuation_chunks=1,
    pad_token_idx=0,
    max_len=None,
) -> RetroDataset:

    spec = json.load(spec_file.open())
    base_dir = spec_file.parent

    # input dataset
    input_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=base_dir / shard["chunks"],
            seq2chunk=base_dir / shard["seq2chunk"],
            chunk2seq=base_dir / shard["chunk2seq"]
        )
        for shard in spec["shards"]
    ])

    # retrieval dataset
    index_spec = json.load((base_dir / spec["neighbours"]["index_spec"]).open())
    index_base_dir = base_dir / Path(spec["neighbours"]["index_spec"]).parent
    retrieval_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=index_base_dir / shard["chunks"],
            seq2chunk=index_base_dir / shard["seq2chunk"],
            chunk2seq=index_base_dir / shard["chunk2seq"]
        )
        for shard in index_spec
    ])

    # neighbour dataset
    neighbour_dataset = ShardedChunkNeighbourDataset([
        ChunkNeighbourDataset(
            neighbours=base_dir / shard["neighbours"],
            retrieval_dataset=retrieval_dataset
        )
        for shard in spec["shards"]
    ])

    retro_dataset = RetroDataset(
        input_dataset=input_dataset,
        neighbour_dataset=neighbour_dataset,
        num_neighbours=num_neighbours,
        continuation_chunks=continuation_chunks,
        pad_token_idx=pad_token_idx,
        max_len=max_len
    )

    return retro_dataset


def retro_collate_fn(batch: List[RetroTrainingExample], pad_token_idx: int):
    max_input_len = max(ex.input_ids.shape[0] for ex in batch)
    max_input_chunks = max(ex.neighbour_ids.shape[0] for ex in batch)
    max_neighbour_len = max(ex.neighbour_ids.shape[-1] for ex in batch)
    
    input_ids = torch.stack([
        F.pad(ex.input_ids, (0, max_input_len - ex.input_ids.shape[0]), value=pad_token_idx)
        for ex in batch
    ])
    neighbour_ids = torch.stack([
        F.pad(ex.neighbour_ids, (0, max_neighbour_len - ex.neighbour_ids.shape[-1], 
                                 0, 0,
                                 0, max_input_chunks - ex.neighbour_ids.shape[0]), value=pad_token_idx)
        for ex in batch
    ])
    labels = torch.stack([
        F.pad(ex.labels, (0, max_input_len - ex.labels.shape[0]), value=-100)
        for ex in batch
    ])
    return input_ids, neighbour_ids, labels



class RetroModelLMHeadLightning(RetroModelLMHead, pl.LightningModule):

    def __init__(self, config: RetroConfig, retriever: Optional[Retriever]=None, device='cpu', retrieval_interval=64):
        super().__init__(config, retriever=retriever, device=device)
        self.val_loss_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()
        self.retrieval_interval = retrieval_interval

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.1)

    def training_step(self, batch, _):
        input_ids, neighbour_ids, labels = batch
        output = self.forward(input_ids, neighbour_ids, labels=labels)
        self.log("train_loss", output.loss)
        return output.loss

    def validation_step(self, batch, batch_idx, *args):
        input_ids, neighbour_ids, labels = batch
        output = self.forward(input_ids,  neighbour_ids, labels=labels, loss_reduction="none")

        # Make sure to reset metric when using multiple dataloaders
        if batch_idx == 0:
            self.val_loss_metric.reset()

        self.val_loss_metric.update(output.loss[labels != -100])
        self.log("val_loss", self.val_loss_metric, on_epoch=True, prog_bar=True)
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log("perplexity", torch.exp(self.val_loss_metric.compute()), on_epoch=True, prog_bar=True)
        return output.loss

    def test_step(self, batch, batch_idx, *args):
        input_ids, neighbour_ids, labels = batch
        output = self.forward(input_ids,  neighbour_ids, retrieval_interval=self.retrieval_interval, labels=labels, loss_reduction="none")

        # Make sure to reset metric when using multiple dataloaders
        if batch_idx == 0:
            self.test_loss_metric.reset()

        self.test_loss_metric.update(output.loss[labels != -100])
        self.log("test_loss", self.test_loss_metric, on_epoch=True, prog_bar=True)
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log("perplexity", torch.exp(self.test_loss_metric.compute()), on_epoch=True, prog_bar=True)
        return output.loss

class RetroModelLMHeadLightningInference(RetroModelLMHeadInference, pl.LightningModule):

    def __init__(self, config: RetroConfig, retriever: Optional[Retriever]=None, device='cpu'):
        super().__init__(config, retriever=retriever, device=device)
        self.val_loss_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()

def main(args):

    config = RetroConfig(**json.load(args.retro_config.open()))
    
    train_ds = get_retro_dataset_from_spec(
        spec_file=args.training_dataset_spec,
        num_neighbours=args.num_neighbours,
        continuation_chunks=args.num_continuation_chunks,
        pad_token_idx=config.pad_token_idx,
        max_len=args.max_len
    )
    if args.training_data_subset_indices:
        train_ds = Subset(train_ds, [int(i) for i in open(args.training_data_subset_indices)])
        print(f"Using subset of training data of size: {len(train_ds)}")

    val_ds = get_retro_dataset_from_spec(
        spec_file=args.validation_dataset_spec,
        num_neighbours=args.num_neighbours,
        continuation_chunks=args.num_continuation_chunks,
        pad_token_idx=config.pad_token_idx,
        max_len=args.max_len
    )
    if args.validation_data_subset_indices:
        val_ds = Subset(val_ds, [int(i) for i in open(args.validation_data_subset_indices)])
        print(f"Using subset of validation data of size: {len(val_ds)}")

    collate_fn = partial(retro_collate_fn, pad_token_idx=config.pad_token_idx)

    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    model = RetroModelLMHeadLightning(config)

    callbacks = []
    if args.experiment_dir is not None:
        args.experiment_dir = args.experiment_dir.absolute()
        logger = TensorBoardLogger(save_dir=str(args.experiment_dir.parent), name=args.experiment_dir.name)
        callbacks.append(ModelCheckpoint())
    else:
        logger = None

    trainer = pl.Trainer(
        default_root_dir=str(args.experiment_dir.parent) if args.experiment_dir is not None else None,
        strategy="ddp_find_unused_parameters_false" if args.gpus_per_node is not None else None,
        gpus=args.gpus_per_node, 
        num_nodes=args.num_nodes,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dl
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--training-dataset-spec", required=True, type=Path)
    parser.add_argument("--validation-dataset-spec", required=True, type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--num-neighbours", type=int)
    parser.add_argument("--num-continuation-chunks", type=int, default=1)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--training-data-subset-indices")
    parser.add_argument("--validation-data-subset-indices")

    # Model args
    parser.add_argument("--retro-config", required=True, type=Path)
    parser.add_argument("--retrofit")

    # Training args
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--val-check-interval", type=int, default=20_000)

    args = parser.parse_args()
    main(args)

