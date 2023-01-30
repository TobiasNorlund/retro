from argparse import ArgumentError
import json
import readline
import torch
from pathlib import Path
from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from modeling_retro import RetroConfig
from sentence_transformers import SentenceTransformer
from retrieval import RetrieverWithCache, IndexServiceRetriever, IndexServiceClient
from train_retro import RetroModelLMHeadLightning
from data.tokenize_and_chunk import get_tokenizer


class DemoRetriever(RetrieverWithCache):
    
    def __init__(self, num_neighbours: int, neighbour_len: int, tokenizer):
        super().__init__(num_neighbours, neighbour_len)
        self.tokenizer = tokenizer

    def get_neighbours_for_chunk(self, chunk: torch.LongTensor):
        """
        chunk - [chunk size]
        Returns:
         neighbours - [num neighbours, neighbour len]
        """
        print("Please input neighbours for the following chunk:")
        print(self.tokenizer.decode(chunk))
        ret = torch.zeros((self.num_neighbours, self.neighbour_len), dtype=torch.int64)
        for neighbour_idx in range(self.num_neighbours):
            neighbour_text = input(f"Neighbour {neighbour_idx}: ")
            encoded_neighbour = self.tokenizer.encode(neighbour_text, return_tensors="pt", add_special_tokens=False)[0, :self.neighbour_len]
            ret[neighbour_idx, :encoded_neighbour.shape[0]] = encoded_neighbour
        print()
        return ret


def main(args):

    config = RetroConfig(**json.load(args.retro_config.open()))
    tokenizer = get_tokenizer()

    retriever = DemoRetriever(
        num_neighbours=args.num_neighbours, 
        neighbour_len=config.chunk_size * (1 + args.num_continuation_chunks),
        tokenizer=tokenizer
    )
        
    model = RetroModelLMHeadLightning.load_from_checkpoint(str(args.checkpoint), config=config, retriever=retriever).eval()
    prompt = args.prompt

    while True:

        if prompt is None:
            print("Input prompt:")
            prompt = input()

        input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
        res = model.generate(
            inputs=input_ids, 
            do_sample=False,
            num_beams=1,
            top_k=5,
            top_p=1,
            temperature=1,
            min_length=10,
            max_length=200,
            length_penalty=1,
            early_stopping=False,
            num_beam_groups=1,
            num_return_sequences=1,
            repetition_penalty=1,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=0,
            diversity_penalty=0.0,
            remove_invalid_values=False,
            pad_token_id=0, 
            eos_token_id=1,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )
        prompt = None

        print("-- Generation complete --")
        print()
        print(tokenizer.decode(res[0]))
        print()
        print("-------------------------")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retro-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--num-neighbours", type=int, default=2)
    parser.add_argument("--num-continuation-chunks", type=int, default=1)

    args = parser.parse_args()

    main(args)