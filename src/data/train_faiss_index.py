import faiss
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path


def main(args):

    spec = json.load(args.spec.open("r"))
    base_dir = args.spec.parent

    # Load up all training embedding shards into contiguous array and convert to float32
    embeddings_shards_memmap = [] 
    total_embeddings = 0
    for shard in spec:
        embeddings_shards_memmap.append(np.load(base_dir / shard["embeddings"], mmap_mode="r"))
        total_embeddings += embeddings_shards_memmap[-1].shape[0]

        if total_embeddings > args.max_training_vectors:
            break
    
    embeddings = np.empty((total_embeddings, embeddings_shards_memmap[0].shape[1]), dtype=np.float32)
    n = 0
    for shard in tqdm(embeddings_shards_memmap, desc="Loading embeddings"):
        embeddings[n:n+shard.shape[0],:] = shard
        n += shard.shape[0]

    # Create index
    index = faiss.index_factory(embeddings.shape[1], args.index_type, faiss.METRIC_L2)
    if args.use_gpus is True:
        co = faiss.GpuMultipleClonerOptions()
        index = faiss.index_cpu_to_all_gpus(index, co)

    # Train index
    print("Training index...")
    index.train(embeddings)

    # Save
    print("Saving index...")
    if args.use_gpus is True:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--index-type", required=True)
    parser.add_argument("--use-gpus", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-training-vectors", type=int, default=np.inf)
    args = parser.parse_args()

    main(args)
