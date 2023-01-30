import torch
import io
import zipfile
import numpy as np
import requests
from abc import ABC, abstractmethod

from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset


class Retriever(ABC):

    @property
    @abstractmethod
    def num_neighbours(self):
        pass

    @property
    @abstractmethod
    def neighbour_len(self):
        pass

    @abstractmethod
    def retrieve_neighbours(self, chunks: torch.LongTensor):
        """
        chunks:     [batch size, chunk size]

        Returns:
          neighbours:  [batch size, num neighbours, neighbour len]
        """
        pass


class RetrieverWithCache(Retriever):

    def __init__(self, num_neighbours: int, neighbour_len: int):
        self._num_neighbours = num_neighbours
        self._neighbour_len = neighbour_len
        self.cached_chunks = None
        self.cached_neighbours = None
    
    @property
    def num_neighbours(self):
        return self._num_neighbours

    @property
    def neighbour_len(self):
        return self._neighbour_len

    @abstractmethod
    def get_neighbours_for_chunk(self, chunk: torch.LongTensor):
        """
        chunk - [chunk size]
        Returns:
         neighbours - [num neighbours, neighbour len]
        """
        raise NotImplementedError()

    def retrieve_neighbours(self, chunks: torch.LongTensor):

        if self.cached_chunks is None and self.cached_neighbours is None:
            self.cached_chunks = torch.zeros((0, chunks.shape[1]), dtype=torch.int64)
            self.cached_neighbours = torch.zeros((0, self.num_neighbours, self.neighbour_len), dtype=torch.int64)

        cache_matches = torch.all(chunks[:, None, :] == self.cached_chunks[None, :, :], dim=-1)

        ret = torch.zeros((chunks.shape[0], self.num_neighbours, self.neighbour_len), dtype=torch.int64)
        for chunk_idx in range(chunks.shape[0]):
            cache_idx = cache_matches[chunk_idx].nonzero()[:, 0]
            if cache_idx.shape[0] == 1:
                ret[chunk_idx, :, :] = self.cached_neighbours[cache_idx[0]]
            else:
                neighbours = self.get_neighbours_for_chunk(chunks[chunk_idx])
                ret[chunk_idx, :, :] = neighbours

                # Add to cache
                self.cached_chunks = torch.concat((self.cached_chunks, chunks[[chunk_idx]]))
                self.cached_neighbours = torch.concat((self.cached_neighbours, neighbours[None, ...]))

        return ret


class DummyRetriever(Retriever):

    def __init__(self, num_neighbours: int):
        self._num_neighbours = num_neighbours
        self._neighbour_len = self.neighbour_len

    @property
    def num_neighbours(self):
        return self._num_neighbours

    @property
    def neighbour_len(self):
        return self._neighbour_len
    
    def retrieve_neighbours(self, chunks: torch.LongTensor):
        return torch.ones(
            (chunks.shape[0], self.num_neighbours, self.neighbour_len), 
            dtype=torch.int64,
            device=chunks.device
        )


class IndexServiceClient:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def is_available(self):
        try:
            _ = requests.get(f"http://{self.host}:{self.port}/health")
            return True
        except:
            return False

    def query(self, embeddings, k: int):
        """
        Query the index with `embeddings`
        """
        buf = io.BytesIO()
        np.save(buf, embeddings)
        resp = requests.post(f"http://{self.host}:{self.port}/query?k={k}", data=buf.getvalue())

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zip:
            distances = np.load(io.BytesIO(zip.read("distances")))
            key_indices = np.load(io.BytesIO(zip.read("indices")))
            key_embeddings = np.load(io.BytesIO(zip.read("embeddings")))

        return distances, key_indices, key_embeddings


class IndexServiceRetriever(RetrieverWithCache):

    def __init__(
        self, 
        index_service: IndexServiceClient,
        retrieval_dataset: ShardedChunkedSequenceDataset,
        retrieval_model,
        tokenizer,
        num_neighbours: int,
        chunk_size: int,
        num_continuation_chunks: int,
        verbose: bool=True
    ):
        super().__init__(num_neighbours, chunk_size * (1 + num_continuation_chunks))
        self.index_service = index_service
        self.retrieval_dataset = retrieval_dataset
        self.tokenizer = tokenizer
        self.retrieval_model = retrieval_model
        self.num_continuation_chunks = num_continuation_chunks
        self.verbose = verbose

    def get_neighbours_for_chunk(self, chunk: torch.LongTensor):
        """
        chunk - [chunk size]
        Returns:
         neighbours - [num neighbours, neighbour len]
        """
        chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
        if self.verbose:
            print("Retrieving neighbours for chunk:")
            print(chunk_text)
            print()
        embedding = self.retrieval_model.encode(
            [chunk_text],
            output_value="sentence_embedding",
            normalize_embeddings=True
        )

        # Retrieve neighbour chunks
        _, neighbour_chunk_indices, _ = \
            self.index_service.query(embedding, k=self.num_neighbours)

        neighbours = torch.zeros((self.num_neighbours, self.neighbour_len), dtype=torch.int64)
        for neighbour_idx, neighbour_chunk_idx in enumerate(neighbour_chunk_indices[0]):
            neighbour_tokens = self.retrieval_dataset.get_chunk_tokens(
                neighbour_chunk_idx, 
                include_continuation_chunks=self.num_continuation_chunks
            )
            if self.verbose:
                print(f"Neighbour {neighbour_idx}:")
                print(self.tokenizer.decode(neighbour_tokens, skip_special_tokens=True))
            neighbours[neighbour_idx, :neighbour_tokens.shape[0]] = torch.tensor(neighbour_tokens.astype(np.int64))
        
        if self.verbose:
            print()

        return neighbours


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from data.tokenize_and_chunk import get_tokenizer
    from pathlib import Path
    import json

    index_service = IndexServiceClient("localhost", 8000)
    assert index_service.is_available(), "The faiss index service is not available"

    # Load retrieval index
    retrieval_index_spec = Path("../data/datasets/MassiveOpenText/retriever_sentence_transformer/val.index.spec.json")
    index_spec = json.load(retrieval_index_spec.open())
    index_base_dir = retrieval_index_spec.parent
    retrieval_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=index_base_dir / shard["chunks"],
            seq2chunk=index_base_dir / shard["seq2chunk"],
            chunk2seq=index_base_dir / shard["chunk2seq"]
        )
        for shard in index_spec
    ])
    
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = get_tokenizer()
    retriever = IndexServiceRetriever(
        index_service=index_service,
        retrieval_dataset=retrieval_dataset,
        retrieval_model=retrieval_model,
        tokenizer=tokenizer,
        num_neighbours=2,
        chunk_size=64,
        num_continuation_chunks=1
    )

    test_chunk_text = "On December 24th is Christmas eve and every child gets presents from Santa Claus."
    test_chunk = tokenizer.encode(test_chunk_text, return_tensors="pt")[0]
    retriever.get_neighbours_for_chunk(
        test_chunk
    )