import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tokenize_and_chunk import get_tokenizer as get_chunk_tokenizer
from tqdm import tqdm


def main(args):
    chunk_tokenizer = get_chunk_tokenizer()
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
     
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        model = model.to(device)
    else:
        device = "cpu"

    print("Loading chunks...")
    chunks = np.load(args.input)

    if args.verbose:
        print("Decoding chunks...")
    decoded_batch_seqs = []
    for i in tqdm(range(chunks.shape[0]), disable=not args.verbose):
        decoded_batch_seqs.append(chunk_tokenizer.decode(chunks[i,:], skip_special_tokens=True))

    if args.verbose:
        print("Embedding...")
    sentence_embeddings = model.encode(
        decoded_batch_seqs,
        batch_size=args.batch_size,
        device=device,
        convert_to_numpy=True,
        output_value="sentence_embedding",
        normalize_embeddings=True,
        show_progress_bar=args.verbose
    )
    sentence_embeddings = np.stack(sentence_embeddings).astype(np.float16)
    np.save(args.output, sentence_embeddings)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Tokenized chunks (*.npy)")
    parser.add_argument("--output", help="Output file name (*.npy)")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--gpu_id", default=None, type=int)
    parser.add_argument("--verbose", default=True, action="store_true")
    args = parser.parse_args()
    main(args)
