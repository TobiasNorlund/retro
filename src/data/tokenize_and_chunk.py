import sys
import json
import numpy as np
import logging
from pathlib import Path
from transformers import AutoTokenizer


def get_tokenizer():
    # Set model_max_length to suppress warning
    return AutoTokenizer.from_pretrained("t5-base", use_fast=True, model_max_length=10000)


def main(args):

    tokenizer = get_tokenizer()
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # Supress long document warning
    
    seq2chunk_index = []  # Index from seq_idx to idx of first chunk in sequence
    chunk2seq_index = []  # Index from chunk to seq_idx
    chunks_buffer = np.empty(shape=(args.max_chunks, args.chunk_size), dtype=np.uint16)
    tot_chunks = 0
    for n_seq, line in enumerate(args.input):
        parsed = json.loads(line)
        tokens = tokenizer.encode(
            parsed["text"], 
            return_tensors="np", 
            add_special_tokens=True,
            padding=True,
            pad_to_multiple_of=args.chunk_size)[0]
        tokens = tokens.reshape(-1, args.chunk_size)
        n_chunks = tokens.shape[0]

        chunks_buffer[tot_chunks:(tot_chunks + n_chunks), :] = tokens
        seq2chunk_index.append(tot_chunks)
        chunk2seq_index += [n_seq] * n_chunks
        tot_chunks += n_chunks

        if args.verbose:
            print(f"\rNum seqs: {n_seq}    Num chunks: {tot_chunks}", end="")

    # Save with correct shape
    np.save(args.chunks_out, chunks_buffer[:tot_chunks, :])
    np.save(args.seq2chunk_index_out, np.array(seq2chunk_index, dtype=np.int64))
    np.save(args.chunk2seq_index_out, np.array(chunk2seq_index, dtype=np.int32))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="File with json lines with key \"text\"")
    parser.add_argument("--chunks-out", type=Path, required=True)
    parser.add_argument("--seq2chunk-index-out", type=Path, required=True)
    parser.add_argument("--chunk2seq-index-out", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)