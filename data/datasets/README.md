## Instructions for creating a custom dataset for training or evaluating RETRO

Here we describe the pipeline of steps required to build a Retro dataset.

First, we assume the dataset is in json lines format, with one document per line, and with the key `"text"`:

```json
{"text": "Here is document 1"}
{"text": "Here is document 2"}
...
```

### 1. Tokenize and chunk data
We start by tokenizing and splitting each document into fixed size chunks, run:

```bash
$ cd /workspace/src/data
$ mkdir -p /workspace/data/datasets/MyDataset
$ cat data.jsonl | python tokenize_and_chunk.py \
    --chunks-out /workspace/data/datasets/MyDataset/chunks.npy \
	--seq2chunk-index-out /workspace/data/datasets/MyDataset/seq2chunk.npy \
	--chunk2seq-index-out /workspace/data/datasets/MyDataset/chunk2seq.npy \
	--chunk-size 64 \
	--max-chunks 10000000 \
	--verbose
```

This will create three files:
 - /workspace/data/datasets/MyDataset/chunks.npy : Array containing tokenized chunks
 - /workspace/data/datasets/MyDataset/seq2chunk.npy : Array indexing chunk starting positions for each document
 - /workspace/data/datasets/MyDataset/chunk2seq.npy : Array indexing document positions for each chunk

### 2. Embed chunks

Next, we precompute embeddings for all chunks:

```bash
$ cd /workspace/src/data
$ python embed_chunks.py \
    /workspace/data/datasets/MyDataset/chunks.npy \
	/workspace/data/datasets/MyDataset/chunks.embeddings.npy \
	--batch-size 256  # Or whatever you can fit in memory
```

This will create Sentence Transformer embeddings for all chunks, and store the resulting embeddings in `chunks.embeddings.npy`.

We support splitting up an index dataset into multiple shards. A config file is used to specify the paths of all files constituting a dataset.
In this example, we only have one shard and create the config file like this:

```bash
$ cat << EOF > /workspace/data/datasets/MyDataset/index.spec.json
[
    {
        "chunks": "chunks.npy",
        "seq2chunk": "seq2chunk.npy",
        "chunk2seq": "chunk2seq.npy",
        "embeddings": "chunks.embeddings.npy"
    }
]
EOF
```

### 3. Train `faiss` index

We will use `faiss` to index and search nearest neighbours for chunks.
For this, we will need to train an index.
Depending on the amount of chunks you can probably optimize the exact index, but we have used `IVF131072,PQ32`.

```bash
$ cd /workspace/src/data
$ python train_faiss_index.py \
	--spec /workspace/data/datasets/MyDataset/index.spec.json \
	--max-training-vectors $((131072 * 256)) \
	--index-type IVF131072,PQ32 \
	--output /workspace/data/datasets/MyDataset/IVF131072,PQ32.index \
	--use-gpus  # Optional, but makes it much faster
```

### 4. Fill trained index with data

With the trained index, we can now add our chunk embeddings to it

```bash
$ cd /workspace/src/data
$ python -u build_faiss_index.py \
	--spec /workspace/data/datasets/MyDataset/index.spec.json \
	--trained-index /workspace/data/datasets/MyDataset/IVF131072,PQ32.index \
	--output-index /workspace/data/datasets/MyDataset/data.index \
	--use-gpus \  # Optional
	--shard-index  # Optional, to shard index on multiple GPUs
```

### 5. Retrieve neighbours using index

Finally, we can use our index to retrieve neighbours to our chunks, to be used for training or evaluation of RETRO.

```bash
$ cd /workspace/src/data
$ python -u ../../src/data/retrieve_neighbours.py \
	--query-embeddings \
		/workspace/data/datasets/MyDataset/chunks.embeddings.npy \
	--query-chunk2seq \
		/workspace/data/datasets/MyDataset/chunk2seq.npy \
	--neighbours-output \
		/workspace/data/datasets/MyDataset/chunks.neighbours.npy \
	--index /workspace/data/datasets/MyDataset/data.index \
	--index-spec /workspace/data/datasets/MyDataset/index.spec.json \
	--num-neighbours 4 \
	--use-gpus \
	--shard-index
```

This will create `chunks.neighbours.npy` that contains indices to the nearest neighbours.


### 6. Create RETRO dataset spec file

For the dataset to be used for training or evaluating RETRO, we need to create another spec file:

```bash
$ cat << EOF > /workspace/data/datasets/MyDataset/data.spec.json
{
	"shards": [
		{
			"chunks": "chunks.npy",
			"seq2chunk": "seq2chunk.npy",
			"chunk2seq": "chunk2seq.npy",
			"neighbours": "chunks.neighbours.npy"
		}
	],
	"neighbours": {
		"faiss_index": "data.index",
		"index_spec": "index.spec.json"
	}
}
EOF
```

The resulting `data.spec.json` can now be fed as `--training-dataset-spec` or `--validation-dataset-spec` to e.g. `train_retro.py`. It can also be fed as `--test-dataset-spec` to `evaluate_retro.py`



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
