import faiss
import json
import numpy as np
from pathlib import Path


def get_valid_neighbours(retrieved_chunk_indices, query_chunk2seq, offset):
    num_chunks_in_shard = query_chunk2seq.shape[0]
    offset_retrieved_chunk_indices = retrieved_chunk_indices - offset
    # off shard neighbours are chunks not part of current queries
    off_shard = (offset_retrieved_chunk_indices < 0) | (offset_retrieved_chunk_indices >= num_chunks_in_shard)
    offset_retrieved_chunk_indices[off_shard] = 0  # Set to 0 to avoid indexing error below
    # invalid neighbours are chunks that are part of same sequence as query chunk, that are also not off shard chunks
    invalid = (query_chunk2seq[offset_retrieved_chunk_indices] == query_chunk2seq[:, None]) & np.logical_not(off_shard)
    retrieved_chunk_indices[invalid] = -1
    valid_neighbours = np.take_along_axis(retrieved_chunk_indices, np.argsort(invalid), -1)
    return valid_neighbours


def main(args):

    print("Load index...")
    index = faiss.read_index(str(args.index))
    assert index.is_trained, "The index must be trained"
    if args.use_gpus:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = args.shard_index
        index = faiss.index_cpu_to_all_gpus(index, co)

    assert len(args.query_embeddings) == len(args.query_chunk2seq) == len(args.neighbours_output)
    for query_embeddings_file, query_chunk2seq_file, neighbours_output_file in zip(args.query_embeddings, args.query_chunk2seq, args.neighbours_output):
        print(f"Processing {query_embeddings_file} ...")
        # If the query embeddings are also indexed, we need to make sure we're not retreiving from same sequence as query
        base_dir = args.index_spec.absolute().parent
        query_index_offset = 0
        for shard in json.load(args.index_spec.open()):
            if Path(query_embeddings_file).samefile(base_dir / shard["embeddings"]):
                print("  The queries seem to be in index. Will need to filter neighbours from same sequence as query")
                query_chunk2seq = np.load(query_chunk2seq_file)
                break
            query_index_offset += np.load(base_dir / shard["embeddings"], mmap_mode="r").shape[0]
        else:
            print("  The queries do not seem to be indexed. No need to filter neighbours from same sequence as query")
            query_index_offset = None

        print("Load query embeddings...")
        query_embeddings = np.load(query_embeddings_file).astype("float32")

        print("Running search...")
        _, retrieved_chunk_indices = index.search(query_embeddings, args.num_neighbours + args.retrieval_margin)

        # Make sure we only retrieve valid neighbours (e.g. not from same sequence as query)
        if query_index_offset is not None:
            print("Filter retrieved neighbours from same sequence as query...")
            retrieved_chunk_indices = get_valid_neighbours(
                retrieved_chunk_indices, 
                query_chunk2seq,
                query_index_offset
            )

        print("Saving retrieved neighbours...")
        np.save(neighbours_output_file, retrieved_chunk_indices[:, :args.num_neighbours])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-embeddings", required=True, nargs="+")
    parser.add_argument("--query-chunk2seq", required=True, nargs="+")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--index-spec", required=True, type=Path)
    parser.add_argument("--num-neighbours", type=int, required=True)
    parser.add_argument("--neighbours-output", required=True, nargs="+")
    parser.add_argument("--retrieval-margin", type=int, default=4, help="Number of extra neighbours to retrieve, in case some are from same seq as query")
    parser.add_argument("--use-gpus", action="store_true")
    parser.add_argument("--shard-index", action="store_true")
    args = parser.parse_args()
    main(args)
