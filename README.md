# PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design

We developed our project based on [this repository](https://github.com/TobiasNorlund/retro).

## Environment

```
conda create -n retro python=3.9 -y
conda activate retro

# if use torch 2.x to use torch.compile (https://pytorch.org/get-started/locally/) 
pip3 install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# if stick to the old version
pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
                torch==1.12.1+cu113 \
                torchvision==0.13.1+cu113 \
                torchaudio==0.12.1+cu113 

# CUDA & cuDNN version must match the onnxruntime version
# Wenqi: it seems that even if for CUDA 12.0, if we install pytorch based on 11.8, it would work, no need to reinstall CUDA!
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements 
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local 

# if installed on AWS AMI, later on ‘Failed to initialize NVML: Driver/library version mismatch’ may appear because the system by default forces another CUDA version
# solution: https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get upgrade # The following packages have unmet dependencies:  nvidia-driver-520 : Depends: nvidia-driver-525 but it is not installed
sudo apt --fix-broken install
# then reboot


# Sometimes it shows that the fabric manager version does not match the Nvidia driver version - in this case we need to update the driver, e.g., 
sudo apt install nvidia-driver-525
sudo systemctl start nvidia-fabricmanager
systemctl status nvidia-fabricmanager.service
/usr/bin/nv-fabricmanager --version
! python -c "import torch; print(torch.cuda.is_available())"
# sudo apt-get install -y cuda-compat-11-8

pip install transformers==4.21.0 
pip install pytorch-lightning==1.7.4 
pip install einops==0.6.0 
pip install pytest==7.2.1 
pip install sentence-transformers==2.2.2 
pip install matplotlib==3.6.3  
pip install seaborn==0.12.2
pip install torchmetrics==0.11.4

pip install onnx==1.15
pip install onnxruntime-gpu==1.16
# pip install onnxconverter-common==1.14.0
pip install grpcio-tools

conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
# or on CPU-only server
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
```

In ~/.bashrc:

```
WORKSPACE=/fsx/PipeRAG
conda activate retro
```


## Model download

Download the [retro.zip](https://chalmersuniversity.box.com/s/d7qijjdyfv6ubdy1ux10syrq4ep3ca6e) and extract it in `data/model` folder.


## Folder Organization

```
├── Dockerfile
├── LICENSE
├── README.md
├── data
│   ├── datasets
│   └── model
├── inference
│   ├── README.md
│   ├── __pycache__
│   ├── demo_retrieval_client.py
│   ├── demo_retrieval_server.py
│   ├── evaluate_rag_performance.py
│   ├── faiss_server.py
│   ├── grpc_test
│   ├── inference_client.py
│   ├── performance
│   ├── performance_model.py
│   ├── proto
│   ├── retrieval_pb2.py
│   ├── retrieval_pb2.pyi
│   ├── retrieval_pb2_grpc.py
│   ├── retriever_client.py
│   ├── test_retrieval_performance.py
│   └── test_sbert_performance.py
├── logs
├── plots
├── src
│   ├── data
│   ├── dataset_retro.py
│   ├── evaluate_retro_realtime_retrieval.py
│   ├── evaluate_staleness_query_doc_similarity.py
│   ├── evaluation_perplexity_all.py
│   ├── evaluation_suite.py
│   ├── generate_retro_greedy.py
│   ├── generate_retro_onnx.py
│   ├── generate_retro_original.py
│   ├── modeling_retro.py
│   ├── modeling_retro_inference.py
│   ├── modeling_retro_original.py
│   ├── onnx_retro_decoder
│   ├── onnx_retro_encoder
│   ├── out_onnx
│   ├── retrieval.py
│   ├── traces
│   ├── train_retro.py
│   └── unused
└── test_funcs
```

### data

This folder stores the models and datasets for evaluation. 

```
├── datasets
│   ├── MassiveOpenText
│   ├── Pile
│   ├── README.md
│   ├── RealNews
│   ├── c4-en
│   ├── generate_index_config.py
│   ├── index.spec.json
│   ├── indexes_c4
│   ├── indexes_mix
│   ├── indexes_realnews
│   ├── indexes_wikipedia
│   ├── process_data.py
│   ├── val_c4
│   ├── val_realnews
│   ├── val_wikipedia
│   ├── wikipedia-downloader
│   └── wikipedia-en
└── model
    ├── README.md
    ├── model.ckpt
    └── retro.json
```

`process_data.py` is an important script to processing the document datasets, encoding them, and indexing them.

### inference

This is the folder containing the scripts for performance evaluation (both inference and retrieval)

Key files are as follows:

```
├── evaluate_rag_performance.py
├── faiss_server.py
├── inference_client.py
├── performance
├── performance_model.py
├── test_retrieval_performance.py
└── test_sbert_performance.py
```

`evaluate_rag_performance.py` is the script used to automatically evaluate all the generation performance, given that the search service is started.

`faiss_server.py` is used to start the vector search service. 

`inference_client.py` is the inference program using ONNX. The modules in this script is invoked by `evaluate_rag_performance.py`. It also contains a model to get the performance model of inference.

`performance` is a folder storing the trained performance models.

`performance_model.py` contains the performance model modules used to predict the maximum nprobe, using the profiling results of the generation model, the retriever, and the SBERT model.

`test_retrieval_performance.py` is the script to model the retrieval performance.

`test_sbert_performance.py` is the script to model SBERT performance.

### src

Stores all the scripts for perplexity evaluation.

```
├── data
│   ├── embed_chunks.py
│   ├── merge_populated_indexes.py
│   ├── populate_faiss_index.py
│   ├── retrieve_neighbours.py
│   ├── tokenize_and_chunk.py
│   └── train_faiss_index.py
├── dataset_retro.py
├── evaluate_retro_realtime_retrieval.py
├── evaluate_staleness_query_doc_similarity.py
├── evaluation_perplexity_all.py
├── evaluation_suite.py
├── generate_retro_greedy.py
├── generate_retro_onnx.py
├── generate_retro_original.py
├── modeling_retro.py
├── modeling_retro_inference.py
├── modeling_retro_original.py
├── onnx_retro_decoder
├── onnx_retro_encoder
├── retrieval.py
├── train_retro.py
```

`data` folder contains some scripts to process data. But instead of using these scripts, the `process_data.py` script in another folder offers more user-friendly implementation of data preprocessing.

#### Evaluating Perplexity and Quality

`evaluation_perplexity_all.py` is an important script used to evaluate the perplexity of all experiments.

`evaluate_retro_realtime_retrieval.py` is used to evaluate the perplexity of a single algorithm setting, it is invoked by `evaluation_perplexity_all.py` and `evaluation_suite.py`.

`evaluate_staleness_query_doc_similarity.py` is used to evaluate the cosine similarity between content retrieved by stale and non-stale query using sentence transformers.

`evaluation_suite.py` is a deprecated script. It was used to evaluate some perplexity numbers. But now `evaluation_perplexity_all.py` offers more comprehensive functionalities.

`dataset_retro.py` specifies various data loader and iterators for perplexity evaluation, given non-stale and stale queries, with various settings.

`train_retro.py` is a top-level abstraction, containing a function `get_realtime_retrieval_retro_dataset_from_spec` that uses the modules in `dataset_retro.py`. 

The perplexity evaluation script invoking order is: `evaluation_perplexity_all.py` -> `evaluate_retro_realtime_retrieval.py` -> `train_retro.py` -> `dataset_retro.py`

#### ONNX processing

`generate_retro_onnx.py` is the script that exports PyTorch model in ONNX format, with an implementation of generation. The ONNX models are stored in the following folders:

```
├── onnx_retro_decoder
├── onnx_retro_encoder
```

`generate_retro_original.py` is the original PyTorch script for generation using HuggingFace.

`generate_retro_greedy.py` is the PyTorch script with our own greedy decoding implementation.


#### Attention Mechanisms

The following scripts specify the model architecture and the attention mechanisms.

```
├── modeling_retro.py
├── modeling_retro_inference.py
├── modeling_retro_original.py
```

`modeling_retro_original.py` is the original RETRO implementation.

`modeling_retro.py` is the PipeRAG attention implementation used for perplexity evaluation.

`modeling_retro_inference.py` is the PipeRAG attention implementation used for fast inference.

### plots

Stores all the performance numbers as well as the plotting scripts for the Figure. 

The following are the recorded performance and perplexity numbers:
```
├── generation_join_perplexity_and_performance_df.pickle
├── generation_performance_df.pickle
├── generation_perplexity_df.pickle
```

The following scripts are important utils:

```
├── join_df.py
└── print_df.py
```

`join_df.py` is used to join the performance and perplexity. It is very important to run this script to get the latest `generation_join_perplexity_and_performance_df.pickle` which will be used for plotting.

`print_df.py` prints the content of a pickle-stored dataframe.


The following scripts are used for plots used in the paper:

```
├── plot_alternative_system_performance.py
├── plot_dynamic_nprobe.py
├── plot_pareto.py
├── plot_pareto_allow_RETRO_flexible_interval.py
├── plot_ppl_db_sizes_paper.py
├── plot_ppl_nprobe_interval_paper.py
```

`plot_alternative_system_performance.py` projects the performance-perplexity trend on future hardware.

`plot_dynamic_nprobe.py` shows the numbers (used in a table) of the performance-perplexity numbers using dynamic (performance-model-driven) retrievals.

`plot_pareto.py` compares the Pareto performance-perplexity curve of PipeRAG and RETRO.

`plot_pareto_allow_RETRO_flexible_interval.py` compares the Pareto performance-perplexity curve of PipeRAG and RETRO that supports flexible retrieval intervals.

`plot_ppl_db_sizes_paper.py` shows the effect of different database sizes. 

`plot_ppl_nprobe_interval_paper.py` shows the effect of `nprobe` and `intervals` on perplexity.

### (Not important) logs

Stores some logs used in the past.

### (Not important) test_funcs

Some test scripts regarding ONNS, SBERT, etc.

