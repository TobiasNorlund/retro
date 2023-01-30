from pathlib import Path
from data.tokenize_and_chunk import get_tokenizer
from train_retro import get_retro_dataset_from_spec


def main(args):

    tokenizer = get_tokenizer()
    retro_dataset = get_retro_dataset_from_spec(
        args.spec,
        num_neighbours=args.num_neighbours,
        continuation_chunks=args.continuation_chunks,
        pad_token_idx=tokenizer.pad_token_id,
        max_len=args.max_len
    )

    print("")
    print(" Input dataset:")
    print(f"  - {retro_dataset.input_dataset.num_sequences} sequences")
    print(f"  - {retro_dataset.input_dataset.num_chunks} chunks")
    print(f"  - {len(retro_dataset.input_dataset.shards)} shard(s)")
    print()
    print(" Neighbour dataset:")
    print(f"  - {len(retro_dataset.neighbour_dataset)} neighbours")
    print(f"  - {len(retro_dataset.neighbour_dataset.shards)} shard(s)")
    print()
    print(" Retrieval dataset:")
    print(f"  - {retro_dataset.neighbour_dataset.shards[0].retrieval_dataset.num_sequences} sequences")
    print(f"  - {retro_dataset.neighbour_dataset.shards[0].retrieval_dataset.num_chunks} chunks")
    print(f"  - {len(retro_dataset.neighbour_dataset.shards[0].retrieval_dataset.shards)} shard(s)")
    print()

    while True:
        seq_idx = int(input("Enter a sequence index: "))
        input_ids, neighbour_ids, labels = retro_dataset[seq_idx]
        
        print()
        print("Shapes:")
        print(f" input_ids:       {list(input_ids.shape)}")
        print(f" neighbour_ids:   {list(neighbour_ids.shape)}")
        print(f" labels:          {list(labels.shape)}")
        print()
        print(f"Input sequence:")
        print(tokenizer.decode(input_ids, skip_special_tokens=True))
        print()

        input_chunks = input_ids.reshape(-1, retro_dataset.input_dataset.chunk_size)

        for chunk_idx, chunk_tokens in enumerate(input_chunks):
            print(f"Input chunk {chunk_idx+1}:")
            print(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
            print()
            for neighbour_idx in range(neighbour_ids.shape[1]):
                print(f"Neighbour {neighbour_idx+1}:   {tokenizer.decode(neighbour_ids[chunk_idx, neighbour_idx, :], skip_special_tokens=True)}")
            print()
            print()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--num-neighbours", type=int)
    parser.add_argument("--continuation-chunks", type=int, default=1)
    parser.add_argument("--max-len", type=int)
    args = parser.parse_args()
    main(args)