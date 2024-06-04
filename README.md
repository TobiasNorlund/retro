# On the Generalization Ability of Retrieval-Enhanced Transformers

This is the official repo to the paper [On the Generalization Ability of Retrieval-Enhanced Transformers](https://arxiv.org/abs/2302.12128).
We release our [RETRO](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens) implementation along with our trained model.
Due to the large size, we can unfortunately not host the data + retrieval index, but provide the code for reproducing from the raw [Pile](https://pile.eleuther.ai/) and [RealNews](https://github.com/rowanz/grover/tree/master/realnews).


## Environment

**All code and commands in this repo should be executed within the provided Docker environment.**
To build and start the container in a terminal, run:

```bash
$ ./start.sh [--gpu]

...

docker-user@91711f9b80b8:/workspace$ 
```

It might take several minutes to build the Docker image the first time.

### VS Code integration

You can alternatively use the `Dev Containers` extension to VS Code. 
Open this folder in VS Code and click "Reopen in container" and VS Code will do the rest.


## Model download

Download the [model.zip](https://drive.google.com/file/d/1R53kuW_6uWDCXamQy2AghgeseaIPsAcB/view?usp=sharing) and extract it in `data/model` folder.


## Usage

To generate from RETRO, run:

```bash
$ cd src/
$ python generate_retro.py \
    --retro-config /workspace/data/model/retro.json \
    --checkpoint /workspace/data/model/model.ckpt \
    --prompt "A retrieval-enhanced language model is" \
    --num-neighbours 1 \
    --num-continuation-chunks 1
```

You will be prompted to input the neighbour chunks throughout the generation.

## Retrieval data

Instructions for creating a custom retrieval dataset or re-building MassiveOpenText are provided in [data/datasets/README.md](data/datasets/README.md).


## Training RETRO

Our RETRO model was trained with the following command, on four nodes with 4 A100 40GB each. You may have to modify the flags depending on your resource availability.

```bash
$ cd src/
$ python train_retro.py \
	--training-dataset-spec ../data/datasets/MassiveOpenText/train_sentence_transformer_neighbours.spec.json \
	--validation-dataset-spec ../data/datasets/MassiveOpenText/val_sentence_transformer_neighbours.spec.json \
	--experiment-dir ../data/model/ \
	--num-neighbours 2 \
	--num-continuation-chunks 1 \
	--max-len 1024 \
	--retro-config ../data/model/retro.json \
	--batch-size 2 \
	--accumulate-grad-batches 4 \
	--gpus-per-node 4 \
	--num-nodes 4
```

## Running tests

To run tests for validating our RETRO model implementation, run:

```bash
$ cd src/
$ pytest
```


## Citation

```
@inproceedings{norlund-etal-2023-generalization,
    title = "On the Generalization Ability of Retrieval-Enhanced Transformers",
    author = "Norlund, Tobias  and
      Doostmohammadi, Ehsan  and
      Johansson, Richard  and
      Kuhlmann, Marco",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.109",
    doi = "10.18653/v1/2023.findings-eacl.109",
    pages = "1485--1493",
}
```
