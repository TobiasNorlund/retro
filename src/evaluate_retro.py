import argparse
import json
import pytorch_lightning as pl
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from modeling_retro import RetroConfig
from train_retro import RetroModelLMHeadLightning, get_retro_dataset_from_spec, retro_collate_fn


def main(args):

    config = RetroConfig(**json.load(args.retro_config.open()))
    
    test_dss = [get_retro_dataset_from_spec(
        spec_file=args.test_dataset_spec[i],
        num_neighbours=args.num_neighbours,
        continuation_chunks=args.num_continuation_chunks,
        pad_token_idx=config.pad_token_idx,
        max_len=args.max_len
    ) for i in range(len(args.test_dataset_spec))]

    collate_fn = partial(retro_collate_fn, pad_token_idx=config.pad_token_idx)

    test_dls = [DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    ) for test_ds in test_dss]

    model = RetroModelLMHeadLightning.load_from_checkpoint(str(args.checkpoint), config=config, strict=False)

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_false" if args.gpus_per_node is not None else None,
        gpus=args.gpus_per_node, 
        num_nodes=args.num_nodes,
        logger=None,
    )

    trainer.test(model, dataloaders=test_dls) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--test-dataset-spec", nargs='+', required=True, type=Path)
    parser.add_argument("--num-neighbours", type=int)
    parser.add_argument("--num-continuation-chunks", type=int, default=1)
    parser.add_argument("--max-len", type=int)

    # Model args
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--retro-config", required=True, type=Path)

    # Training args
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    main(args)

