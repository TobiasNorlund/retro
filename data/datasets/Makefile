SHELL := /bin/bash

#
# Pile
# -------

Pile/%:
	mkdir -p Pile
	# You need to manually download the data from https://the-eye.eu/public/AI/pile/ and put it in `Pile/`
	exit 1

# 
# RealNews
# -------

RealNews/%.jsonl:
	tar xzf RealNews/realnews.tar.gz realnews/realnews.jsonl -O | tee \
		>(jq -c 'select(.split=="train")' > RealNews/train_realnews.jsonl) \
		>(jq -c 'select(.split=="val")' | head -16400 - > RealNews/val_realnews.jsonl) \
		> /dev/null
	split -l 1040000 --numeric-suffixes=0 --suffix-length=2 --additional-suffix=_realnews.jsonl RealNews/train_realnews.jsonl RealNews/

# 
# MassiveOpenText
# -------

#  Chunks
MassiveOpenText/chunk_shards/%.chunks.npy MassiveOpenText/chunk_shards/%.seq2chunk.npy MassiveOpenText/chunk_shards/%.chunk2seq.npy:
	mkdir -p MassiveOpenText/chunk_shards
	f_name=$$(echo $*) && \
	if [ "$${f_name}" == *"realnews"* ]; then \
			echo HEJ && \
	        python ../../src/data/tokenize_and_chunk.py RealNews/$*.jsonl \
			--chunks-out MassiveOpenText/chunk_shards/$*.chunks.npy \
			--seq2chunk-index-out MassiveOpenText/chunk_shards/$*.seq2chunk.npy \
			--chunk2seq-index-out MassiveOpenText/chunk_shards/$*.chunk2seq.npy \
			--max-chunks 30000000 \
			--verbose; \
	else \
		shard_file=$$(echo $* | cut -f1 -d'_') && \
		if [ -f Pile/train/$${shard_file}.jsonl.zst ]; \
			then unzstd -c Pile/train/$${shard_file}.jsonl.zst; \
			else unzstd -c Pile/$${shard_file}.jsonl.zst; fi | tee \
		>(jq -c 'select(.meta.pile_set_name=="Github")' | \
			python ../../src/data/tokenize_and_chunk.py \
				--chunks-out MassiveOpenText/chunk_shards/$${shard_file}_github.chunks.npy \
				--seq2chunk-index-out MassiveOpenText/chunk_shards/$${shard_file}_github.seq2chunk.npy \
				--chunk2seq-index-out MassiveOpenText/chunk_shards/$${shard_file}_github.chunk2seq.npy \
				--max-chunks 100000000 \
				--verbose) \
		>(jq -c 'select(.meta.pile_set_name=="Wikipedia (en)")' | \
			python ../../src/data/tokenize_and_chunk.py \
				--chunks-out MassiveOpenText/chunk_shards/$${shard_file}_wikipedia_en.chunks.npy \
				--seq2chunk-index-out MassiveOpenText/chunk_shards/$${shard_file}_wikipedia_en.seq2chunk.npy \
				--chunk2seq-index-out MassiveOpenText/chunk_shards/$${shard_file}_wikipedia_en.chunk2seq.npy \
				--max-chunks 100000000 \
				--verbose) \
		>(jq -c 'select(.meta.pile_set_name=="Pile-CC")' | \
			python ../../src/data/tokenize_and_chunk.py \
				--chunks-out MassiveOpenText/chunk_shards/$${shard_file}_pile_cc.chunks.npy \
				--seq2chunk-index-out MassiveOpenText/chunk_shards/$${shard_file}_pile_cc.seq2chunk.npy \
				--chunk2seq-index-out MassiveOpenText/chunk_shards/$${shard_file}_pile_cc.chunk2seq.npy \
				--max-chunks 100000000 \
				--verbose) \
		>(jq -c 'select(.meta.pile_set_name=="Books3")' | \
			python ../../src/data/tokenize_and_chunk.py \
				--chunks-out MassiveOpenText/chunk_shards/$${shard_file}_books3.chunks.npy \
				--seq2chunk-index-out MassiveOpenText/chunk_shards/$${shard_file}_books3.seq2chunk.npy \
				--chunk2seq-index-out MassiveOpenText/chunk_shards/$${shard_file}_books3.chunk2seq.npy \
				--max-chunks 100000000 \
				--verbose) \
		> /dev/null; \
	fi


# Retrieval using Sentence Transformer

#  Chunk embeddings
MassiveOpenText/retriever_sentence_transformer/embeddings/%.embeddings.npy: | MassiveOpenText/chunk_shards/%.chunks.npy 
	python ../../src/data/embed_chunks.py \
		MassiveOpenText/chunk_shards/$*.chunks.npy \
		MassiveOpenText/retriever_sentence_transformer/embeddings/$*.embeddings.npy \
		--batch-size 256 

#  Trained (but empty) index
MassiveOpenText/retriever_sentence_transformer/IVF131072,PQ32.index: | MassiveOpenText/retriever_sentence_transformer/embeddings/00_books3.embeddings.npy MassiveOpen/retriever_sentence_transformer/embeddings/00_realnews.embeddings.npy 
	python ../../src/data/train_faiss_index.py \
		--spec MassiveOpenText/retriever_sentence_transformer/train.index.spec.json \
		--max-training-vectors $$((131072 * 256)) \
		--index-type IVF131072,PQ32 \
		--output MassiveOpenText/retriever_sentence_transformer/IVF131072,PQ32.index \
		--use-gpus

#  Training set index (containing only training chunks)
MassiveOpenText/retriever_sentence_transformer/train.index: | MassiveOpenText/retriever_sentence_transformer/IVF131072,PQ32.index
	python -u ../../src/data/build_faiss_index.py \
		--spec MassiveOpenText/retriever_sentence_transformer/train.index.spec.json \
		--trained-index MassiveOpenText/retriever_sentence_transformer/IVF131072,PQ32.index \
		--output-index MassiveOpenText/retriever_sentence_transformer/train.index \
		--use-gpus \
		--shard-index

#  Validation set index (containing training + val chunks)
MassiveOpenText/retriever_sentence_transformer/val.index: | MassiveOpenText/retriever_sentence_transformer/IVF131072,PQ32.index
	python -u ../../src/data/build_faiss_index.py \
		--spec MassiveOpenText/retriever_sentence_transformer/val.index.spec.json \
		--trained-index MassiveOpenText/retriever_sentence_transformer/IVF131072,PQ32.index \
		--output-index MassiveOpenText/retriever_sentence_transformer/val.index \
		--use-gpus \
		--shard-index
