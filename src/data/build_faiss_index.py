import faiss
import json
import numpy as np
from queue import Queue
from pathlib import Path
from typing import List
from threading import Thread
from time import perf_counter
from contextlib import contextmanager


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def load_embeddings(shard_paths: List[Path], queue: Queue):
    for path in shard_paths:
        queue.put(np.load(path).astype("float32"))


def main(args):

    # Load empty (but trained) index
    print("Loading index...")
    index = faiss.read_index(str(args.trained_index))
    assert index.is_trained, "The index must be trained"

    if args.use_gpus:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = args.shard_index
        index = faiss.index_cpu_to_all_gpus(index, co)
    
    print("Adding embedding shards...")
    spec = json.load(args.spec.open("r"))
    base_dir = args.spec.parent
    shard_paths = [base_dir / shard["embeddings"] for shard in spec]

    # Start background thread to load embedding shards from disk
    embeddings_queue = Queue(maxsize=3)
    worker = Thread(target=load_embeddings, args=(shard_paths, embeddings_queue))
    worker.start()

    num_embs_added = 0
    for i in range(len(shard_paths)):
        embs = embeddings_queue.get()
        index.add(embs)
        
        # Evaluate after adding each shard
        with catchtime() as t:
            _, I = index.search(embs[:args.num_queries,:], 1)
        qps = args.num_queries / t()
        recall_at_1 = np.mean(I[:,0] == (np.arange(args.num_queries) + num_embs_added))
        num_embs_added += embs.shape[0]
        
        print(f"{i+1}:\t{num_embs_added} vectors \tRecall@1: {recall_at_1} \tQueries / s: {int(qps)}")

    print("Saving index...")
    if args.use_gpus is True:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, str(args.output_index))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--trained-index", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--use-gpus", action="store_true")
    parser.add_argument("--shard-index", action="store_true")
    parser.add_argument("--num-queries", default=100000, type=int)
    args = parser.parse_args()

    main(args)