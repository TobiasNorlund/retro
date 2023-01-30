## Instructions for creating a custom retrieval dataset

TODO

## Instructions for re-building MassiveOpenText

**Note**: Substantial computational and storage resources are required to prepare MassiveOpenText.

1. Download the uncompressed Pile data file tree from https://mystic.the-eye.eu/public/AI/pile/ and put it in `Pile/`.
2. Request and download the RealNews dataset, and put the `realnews.tar.gz` in `RealNews/`.
3. Tokenize, shard and chunk the data:

```bash
make MassiveOpenText/chunk_shards/{00..29}_wikipedia_en.chunks.npy  # Will also make the other categories (except realnews)
make MassiveOpenText/chunk_shards/{00..29}_realnews.chunks.npy
make MassiveOpenText/chunk_shards/val_wikipedia_en.chunks.npy
make MassiveOpenText/chunk_shards/val_realnews.chunks.npy
```

4. Embed all chunks using Sentence Transformer

```bash
# Embed training set
make MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_books3.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_wikipedia_en.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_github.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_pile_cc.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_realnews.embeddings.npy

# Embed validation set
make MassiveOpenText/retriever_sentence_transformer/embeddings/val_books3.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/val_wikipedia_en.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/val_github.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/val_pile_cc.embeddings.npy
make MassiveOpenText/retriever_sentence_transformer/embeddings/val_realnews.embeddings.npy
```

5. Train a faiss index (requires GPUs)

```bash
make MassiveOpenText/retriever_sentence_transformer/IVF131072,PQ32.index
```

6. Index training set (requires GPUs)

```bash
MassiveOpenText/retriever_sentence_transformer/train.index
```

7. Index validation set (requires GPUs)

```bash
MassiveOpenText/retriever_sentence_transformer/val.index
```

8. Pre-compute neighbours (requires GPUs)

Retrieve neighbours for training set index (containing only training chunks):

```bash
python -u ../../src/data/retrieve_neighbours.py \
	--query-embeddings \
		MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_books3.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_github.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_pile_cc.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_realnews.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/{00..29}_wikipedia_en.embeddings.npy \
	--query-chunk2seq \
		MassiveOpenText/chunk_shards/{00..29}_books3.chunk2seq.npy \
		MassiveOpenText/chunk_shards/{00..29}_github.chunk2seq.npy \
		MassiveOpenText/chunk_shards/{00..29}_pile_cc.chunk2seq.npy \
		MassiveOpenText/chunk_shards/{00..29}_realnews.chunk2seq.npy \
		MassiveOpenText/chunk_shards/{00..29}_wikipedia_en.chunk2seq.npy \
	--neighbours-output \
		MassiveOpenText/retriever_sentence_transformer/neighbours/{00..29}_books3.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/{00..29}_github.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/{00..29}_pile_cc.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/{00..29}_realnews.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/{00..29}_wikipedia_en.neighbours.npy \
	--index MassiveOpenText/retriever_sentence_transformer/train.index \
	--index-spec MassiveOpenText/retriever_sentence_transformer/train.index.spec.json \
	--num-neighbours 4 \
	--use-gpus \
	--shard-index
```

Retrieve neighbours for validation set index (containing training + val chunks):

```bash
python -u ../../src/data/retrieve_neighbours.py \
	--query-embeddings \
		MassiveOpenText/retriever_sentence_transformer/embeddings/val_books3.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/val_github.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/val_pile_cc.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/val_realnews.embeddings.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/val_wikipedia_en.embeddings.npy \
	--query-chunk2seq \
		MassiveOpenText/chunk_shards/val_books3.chunk2seq.npy \
		MassiveOpenText/chunk_shards/val_github.chunk2seq.npy \
		MassiveOpenText/chunk_shards/val_pile_cc.chunk2seq.npy \
		MassiveOpenText/chunk_shards/val_realnews.chunk2seq.npy \
		MassiveOpenText/chunk_shards/val_wikipedia_en.chunk2seq.npy \
	--neighbours-output \
		MassiveOpenText/retriever_sentence_transformer/neighbours/val_books3.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/val_github.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/val_pile_cc.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/val_realnews.neighbours.npy \
		MassiveOpenText/retriever_sentence_transformer/neighbours/val_wikipedia_en.neighbours.npy \
	--index MassiveOpenText/retriever_sentence_transformer/train.index \
	--index-spec MassiveOpenText/retriever_sentence_transformer/train.index.spec.json \
	--num-neighbours 4 \
	--use-gpus \
	--shard-index
```
