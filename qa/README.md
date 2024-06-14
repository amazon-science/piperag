# Run the experiments

Commands:

=== No retrieval ===

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 0 \
	--no-retriever 1 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_no_retrieval_len_128.jsonl

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json \
	--min_len 256 \
	--max_len 256 \
	--use-stale 0 \
	--no-retriever 1 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_no_retrieval_len_256.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json \
	--min_len 512 \
	--max_len 512 \
	--use-stale 0 \
	--no-retriever 1 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_no_retrieval_len_512.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json \
	--min_len 1024 \
	--max_len 1024 \
	--use-stale 0 \
	--no-retriever 1 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_no_retrieval_len_1024.jsonl

=== Wiki ===

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_wikipedia/wikipedia_chunk_0_to_8/IVF1024,PQ32_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_wikipedia/wikipedia_chunk_0_to_8/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 0 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_wiki_retrieval.jsonl

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_wikipedia/wikipedia_chunk_0_to_8/IVF1024,PQ32_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_wikipedia/wikipedia_chunk_0_to_8/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 1 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_wiki_retrieval_stale.jsonl

=== Tiny C4 ===

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 0 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_tiny_c4_retrieval.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/IVF1024,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_1/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 1 \
	--target-total-lines 100 \
	--out-file ./generated_answers/generated_answers_RETRO_tiny_c4_retrieval_stale.jsonl

=== Full C4 ===

* C4 normal, large search space

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_128.jsonl

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 256 \
	--max_len 256 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_256.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 512 \
	--max_len 512 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_512.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 1024 \
	--max_len 1024 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_1024.jsonl


* C4 normal, small search space (nprobe=1)

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 1 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_128_nprobe_1.jsonl

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 256 \
	--max_len 256 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 1 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_256_nprobe_1.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 512 \
	--max_len 512 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 1 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_512_nprobe_1.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 1024 \
	--max_len 1024 \
	--use-stale 0 \
	--target-total-lines 100 \
	--nprobe 1 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_len_1024_nprobe_1.jsonl

* C4 stale, large search space

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 128 \
	--max_len 128 \
	--use-stale 1 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_stale_len_128.jsonl

time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 256 \
	--max_len 256 \
	--use-stale 1 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_stale_len_256.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 512 \
	--max_len 512 \
	--use-stale 1 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_stale_len_512.jsonl


time python generate_answer_using_faiss.py \
    --retro-config $WORKSPACE/data/model/retro.json \
    --checkpoint $WORKSPACE/data/model/model.ckpt \
    --num-neighbours 2 \
    --num-continuation-chunks 1 \
	--index_dir $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/IVF65536,PQ64_populated.index \
	--spec_file $WORKSPACE/data/datasets/indexes_c4/c4_chunk_0_to_999/index.spec.json \
	--min_len 1024 \
	--max_len 1024 \
	--use-stale 1 \
	--target-total-lines 100 \
	--nprobe 16 \
	--out-file ./generated_answers/generated_answers_RETRO_full_c4_retrieval_stale_len_1024.jsonl

# Evaluate the generation quality

Retro (Full C4): 

python eval_na_quality.py --fname generated_answers/generated_answers_RETRO_full_c4_retrieval_len_128.jsonl

128: Average Recall: 0.13073809523809524
256: Average Recall: 0.15073809523809525
512: Average Recall: 0.16407142857142853
Average Recall: 0.17323809523809522


PipeRAG (Full C4): 

python eval_na_quality.py --fname generated_answers/generated_answers_RETRO_full_c4_retrieval_stale_len_128.jsonl

128: Average Recall: 0.13073809523809524
256: Average Recall: 0.14823809523809525
512: Average Recall: 0.16383333333333333
1024: Average Recall: 0.1705

No retrieval: 

python eval_na_quality.py --fname generated_answers/generated_answers_RETRO_no_retrieval_len_128.jsonl 

128: Average Recall: 0.08780952380952384
256: Average Recall: 0.09883333333333334
512: Average Recall: 0.10550000000000001
1024: Average Recall: 0.1155