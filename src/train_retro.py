import argparse
import torch
import torch.nn.functional as F
import json
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from typing import List, Optional
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from modeling_retro import RetroModelLMHead, RetroConfig
from dataset_retro import RetroDataset, ChunkedSequenceDataset, RetroTrainingExample, ShardedChunkedSequenceDataset, \
    ChunkNeighbourDataset, ShardedChunkNeighbourDataset
from retrieval import Retriever


def get_retro_dataset_from_spec(
    spec_file: Path, 
    num_neighbours=None,
    continuation_chunks=1,
    pad_token_idx=0,
    max_len=None,
) -> RetroDataset:

    spec = json.load(spec_file.open())
    base_dir = spec_file.parent

    # input dataset
    input_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=base_dir / shard["chunks"],
            seq2chunk=base_dir / shard["seq2chunk"],
            chunk2seq=base_dir / shard["chunk2seq"]
        )
        for shard in spec["shards"]
    ])

    # retrieval dataset
    index_spec = json.load((base_dir / spec["neighbours"]["index_spec"]).open())
    index_base_dir = base_dir / Path(spec["neighbours"]["index_spec"]).parent
    retrieval_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=index_base_dir / shard["chunks"],
            seq2chunk=index_base_dir / shard["seq2chunk"],
            chunk2seq=index_base_dir / shard["chunk2seq"]
        )
        for shard in index_spec
    ])

    # neighbour dataset
    neighbour_dataset = ShardedChunkNeighbourDataset([
        ChunkNeighbourDataset(
            neighbours=base_dir / shard["neighbours"],
            retrieval_dataset=retrieval_dataset
        )
        for shard in spec["shards"]
    ])

    retro_dataset = RetroDataset(
        input_dataset=input_dataset,
        neighbour_dataset=neighbour_dataset,
        num_neighbours=num_neighbours,
        continuation_chunks=continuation_chunks,
        pad_token_idx=pad_token_idx,
        max_len=max_len
    )

    return retro_dataset


def retro_collate_fn(batch: List[RetroTrainingExample], pad_token_idx: int):
    max_input_len = max(ex.input_ids.shape[0] for ex in batch)
    max_input_chunks = max(ex.neighbour_ids.shape[0] for ex in batch)
    max_neighbour_len = max(ex.neighbour_ids.shape[-1] for ex in batch)
    
    input_ids = torch.stack([
        F.pad(ex.input_ids, (0, max_input_len - ex.input_ids.shape[0]), value=pad_token_idx)
        for ex in batch
    ])
    neighbour_ids = torch.stack([
        F.pad(ex.neighbour_ids, (0, max_neighbour_len - ex.neighbour_ids.shape[-1], 
                                 0, 0,
                                 0, max_input_chunks - ex.neighbour_ids.shape[0]), value=pad_token_idx)
        for ex in batch
    ])
    labels = torch.stack([
        F.pad(ex.labels, (0, max_input_len - ex.labels.shape[0]), value=-100)
        for ex in batch
    ])
    return input_ids, neighbour_ids, labels



class RetroModelLMHeadLightning(RetroModelLMHead, pl.LightningModule):

    def __init__(self, config: RetroConfig, retriever: Optional[Retriever]=None):
        super().__init__(config, retriever)
        self.val_loss_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.1)

    def training_step(self, batch, _):
        input_ids, neighbour_ids, labels = batch
        output = self.forward(input_ids, neighbour_ids, labels=labels)
        self.log("train_loss", output.loss)
        return output.loss

    def validation_step(self, batch, batch_idx, *args):
        input_ids, neighbour_ids, labels = batch
        output = self.forward(input_ids,  neighbour_ids, labels=labels, loss_reduction="none")

        # Make sure to reset metric when using multiple dataloaders
        if batch_idx == 0:
            self.val_loss_metric.reset()

        self.val_loss_metric.update(output.loss[labels != -100])
        self.log("val_loss", self.val_loss_metric, on_epoch=True, prog_bar=True)
        return output.loss

    def test_step(self, batch, batch_idx, *args):
        input_ids, neighbour_ids, labels = batch
        output = self.forward(input_ids,  neighbour_ids, labels=labels, loss_reduction="none")

        # Make sure to reset metric when using multiple dataloaders
        if batch_idx == 0:
            self.test_loss_metric.reset()

        self.test_loss_metric.update(output.loss[labels != -100])
        self.log("test_loss", self.test_loss_metric, on_epoch=True, prog_bar=True)
        return output.loss


def main(args):

    config = RetroConfig(**json.load(args.retro_config.open()))
    
    train_ds = get_retro_dataset_from_spec(
        spec_file=args.training_dataset_spec,
        num_neighbours=args.num_neighbours,
        continuation_chunks=args.num_continuation_chunks,
        pad_token_idx=config.pad_token_idx,
        max_len=args.max_len
    )
    if args.training_data_subset_indices:
        train_ds = Subset(train_ds, [int(i) for i in open(args.training_data_subset_indices)])
        print(f"Using subset of training data of size: {len(train_ds)}")

    val_ds = get_retro_dataset_from_spec(
        spec_file=args.validation_dataset_spec,
        num_neighbours=args.num_neighbours,
        continuation_chunks=args.num_continuation_chunks,
        pad_token_idx=config.pad_token_idx,
        max_len=args.max_len
    )
    if args.validation_data_subset_indices:
        val_ds = Subset(val_ds, [int(i) for i in open(args.validation_data_subset_indices)])
        print(f"Using subset of validation data of size: {len(val_ds)}")

    collate_fn = partial(retro_collate_fn, pad_token_idx=config.pad_token_idx)

    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    model = RetroModelLMHeadLightning(config)

    callbacks = []
    if args.experiment_dir is not None:
        args.experiment_dir = args.experiment_dir.absolute()
        logger = TensorBoardLogger(save_dir=str(args.experiment_dir.parent), name=args.experiment_dir.name)
        callbacks.append(ModelCheckpoint())
    else:
        logger = None

    trainer = pl.Trainer(
        default_root_dir=str(args.experiment_dir.parent) if args.experiment_dir is not None else None,
        strategy="ddp_find_unused_parameters_false" if args.gpus_per_node is not None else None,
        gpus=args.gpus_per_node, 
        num_nodes=args.num_nodes,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dl
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--training-dataset-spec", required=True, type=Path)
    parser.add_argument("--validation-dataset-spec", required=True, type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--num-neighbours", type=int)
    parser.add_argument("--num-continuation-chunks", type=int, default=1)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--training-data-subset-indices")
    parser.add_argument("--validation-data-subset-indices")

    # Model args
    parser.add_argument("--retro-config", required=True, type=Path)
    parser.add_argument("--retrofit")

    # Training args
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--val-check-interval", type=int, default=20_000)

    args = parser.parse_args()
    main(args)

