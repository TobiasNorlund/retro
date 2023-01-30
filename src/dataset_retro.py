import numpy as np
import torch
from typing import List
from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset


class ChunkedSequenceDataset:

    def __init__(self, chunks: Path, seq2chunk: Path, chunk2seq: Path):
        self.chunks_path = chunks
        self.chunks = np.load(str(chunks), mmap_mode="r")
        self.seq2chunk = np.load(str(seq2chunk), mmap_mode="r")
        self.chunk2seq = np.load(str(chunk2seq), mmap_mode="r")

    @property
    def chunk_size(self):
        return self.chunks.shape[1]

    @property
    def num_chunks(self):
        return self.chunks.shape[0]

    @property
    def num_sequences(self):
        return self.seq2chunk.shape[0]

    def get_chunk_indices_of_sequence(self, sequence_index):
        chunk_start_idx = self.seq2chunk[sequence_index]
        if sequence_index + 1 < self.seq2chunk.shape[0]:
            chunk_end_idx = self.seq2chunk[sequence_index + 1]
        else:
            # this is the last sequence in the shard
            chunk_end_idx = self.chunks.shape[0]
        return np.arange(chunk_start_idx, chunk_end_idx)

    def get_chunk_tokens(self, chunk_index, include_continuation_chunks=0):
        start_idx = chunk_index
        end_idx = chunk_index + 1
        while end_idx - start_idx - 1 < include_continuation_chunks and \
            end_idx < len(self.chunk2seq) and \
            self.chunk2seq[start_idx] == self.chunk2seq[end_idx]:
            end_idx += 1
        return self.chunks[start_idx:end_idx, :].reshape(-1)


class ShardedChunkedSequenceDataset:

    def __init__(self, shards: List[ChunkedSequenceDataset]):
        self.shards = shards
        assert all(shard.chunk_size == shards[0].chunk_size for shard in shards), \
            "All shards must have same chunk size"

        self.shard_seq_ranges = []
        self.shard_chunk_ranges = []
        self.total_num_chunks = 0
        self.total_num_sequences = 0
        for shard in shards:
            self.shard_seq_ranges.append(range(self.total_num_sequences, self.total_num_sequences + shard.num_sequences))
            self.shard_chunk_ranges.append(range(self.total_num_chunks, self.total_num_chunks + shard.num_chunks))
            self.total_num_sequences += shard.num_sequences
            self.total_num_chunks += shard.num_chunks

    @property
    def chunk_size(self):
        return self.shards[0].chunk_size

    @property
    def num_chunks(self):
        return self.total_num_chunks

    @property
    def num_sequences(self):
        return self.total_num_sequences

    def get_chunk_indices_of_sequence(self, sequence_index):
        for shard_seq_range, shard_chunk_range, shard in zip(self.shard_seq_ranges, self.shard_chunk_ranges, self.shards):
            if int(sequence_index) in shard_seq_range:
                local_seq_index = sequence_index - shard_seq_range.start
                return shard_chunk_range.start + shard.get_chunk_indices_of_sequence(local_seq_index)
        raise IndexError(f"Sequence with index {sequence_index} not found in index")

    def get_chunk_tokens(self, chunk_index, include_continuation_chunks: int=0):
        for shard_range, shard in zip(self.shard_chunk_ranges, self.shards):
            if int(chunk_index) in shard_range:
                local_chunk_index = chunk_index - shard_range.start
                return shard.get_chunk_tokens(local_chunk_index, include_continuation_chunks)
        raise IndexError(f"Chunk with index {chunk_index} not found in index")


class ChunkNeighbourDataset:

    def __init__(self, neighbours: Path, retrieval_dataset: ShardedChunkedSequenceDataset):
        self.neighbours = np.load(str(neighbours), mmap_mode="r")
        self.retrieval_dataset = retrieval_dataset

    @property
    def chunk_size(self):
        return self.retrieval_dataset.chunk_size

    def __len__(self):
        return self.neighbours.shape[0]

    def get_neighbours(self, chunk_index: int, num_neighbours: int=None, continuation_chunks: int=1):
        """
        Returns precomputed tokens for all neighbours of chunk.
        Shape: [num_neighbours, chunk_size * (1 + continuation_chunks)]
        """
        return [
            self.retrieval_dataset.get_chunk_tokens(
                neighbour_chunk_idx, 
                include_continuation_chunks=continuation_chunks
            ) if neighbour_chunk_idx != -1 else None
            for neighbour_chunk_idx in self.neighbours[chunk_index][:num_neighbours]
        ]
            

class ShardedChunkNeighbourDataset:

    def __init__(self, shards: List[ChunkNeighbourDataset]):
        self.shards = shards
        assert all(shard.chunk_size == shards[0].chunk_size for shard in shards), \
            "The chunk size in all shards must match"

        self.shard_ranges = []
        self.total = 0
        for shard in shards:
            self.shard_ranges.append(range(self.total, self.total + len(shard)))
            self.total += len(shard)

    @property
    def chunk_size(self):
        return self.shards[0].chunk_size

    def __len__(self):
        return self.total

    def get_neighbours(self, chunk_index: int, num_neighbours: int=None, continuation_chunks: int=1):
        for shard_range, shard in zip(self.shard_ranges, self.shards):
            if int(chunk_index) in shard_range:
                local_index = chunk_index - shard_range.start
                return shard.get_neighbours(local_index, num_neighbours, continuation_chunks)
        raise IndexError(f"Neighbours for index {chunk_index} not found")


RetroTrainingExample = namedtuple("RetroTrainingExample", [
    "input_ids", 
    "neighbour_ids", 
    "labels"
])

class RetroDataset(Dataset):

    def __init__(
        self, 
        input_dataset: ShardedChunkedSequenceDataset, 
        neighbour_dataset: ShardedChunkNeighbourDataset, 
        num_neighbours=None, 
        continuation_chunks=1, 
        pad_token_idx=0,
        max_len=None
    ):
        super().__init__()
        self.input_dataset = input_dataset
        self.neighbour_dataset = neighbour_dataset
        self.num_neighbours = num_neighbours
        self.continuation_chunks = continuation_chunks
        self.neighbour_size = neighbour_dataset.chunk_size * (1 + continuation_chunks)
        self.pad_token_idx = pad_token_idx
        self.max_num_chunks = max_len // input_dataset.chunk_size if max_len is not None else None

        if max_len is not None:
            assert max_len % input_dataset.chunk_size == 0, \
                "max_len must be a multiple of chunk_size"

        assert input_dataset.num_chunks == len(neighbour_dataset), \
            "The number of chunks in input dataset did not match the number of chunks in neighbour dataset"

    def __len__(self):
        return self.input_dataset.num_sequences

    def __getitem__(self, seq_index: int) -> RetroTrainingExample:
        input_chunk_indices = self.input_dataset.get_chunk_indices_of_sequence(seq_index)
        
        # input_ids
        input_ids = np.concatenate([
            self.input_dataset.get_chunk_tokens(chunk_index)
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # neighbour_ids
        neighbour_ids = np.stack([
            [
                np.pad(neighbour_tokens, (0, self.neighbour_size - len(neighbour_tokens)), constant_values=self.pad_token_idx) \
                    if neighbour_tokens is not None else \
                np.ones(self.neighbour_size) * self.pad_token_idx

                for neighbour_tokens in self.neighbour_dataset.get_neighbours(
                    chunk_index, 
                    num_neighbours=self.num_neighbours, 
                    continuation_chunks=self.continuation_chunks
                )
            ]
            for chunk_index in input_chunk_indices[:self.max_num_chunks]
        ])

        # labels - set to -100 at padded tokens
        labels = np.pad(input_ids[1:], (0, 1), constant_values=self.pad_token_idx).astype(np.int64)
        labels[labels == self.pad_token_idx] = -100

        return RetroTrainingExample(
            torch.from_numpy(input_ids.astype(np.int32)), 
            torch.from_numpy(neighbour_ids.astype(np.int32)), 
            torch.from_numpy(labels)
        )
