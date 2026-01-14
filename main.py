import os
import yaml
import torch
import argparse
import json

# Optional NCCL-related environment settings for distributed training.
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_TREE_THRESHOLD", "0")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from model.tool.regularization import init_regularizer, RegWeightScheduler
from model.transformer_trainer import TransformerTrainer
from model.transformer_tester import TransformerTester
from model.utils_train import (
    init_distributed_mode,
    get_model,
    get_dataloader,
    get_loss,
    set_seed_from_config,
    get_optim_scheduler,
    set_seed,
    restore_model,
    merge_args_into_config,
)


def _privacy_not_available(feature_name: str) -> None:
    raise RuntimeError(
        f"{feature_name} is not available in this anonymized release. "
        "It depends on a proprietary production data/IO pipeline that is intentionally omitted."
    )


# Proprietary online table IO (omitted in anonymized release)
# The actual implementation depends on a proprietary data platform.
try:
    from model.tool.utils_table_io import get_table, create_table_writer  # type: ignore
except Exception:  # pragma: no cover
    def get_table(*args, **kwargs):  # type: ignore
        _privacy_not_available("Online table reader")

    def create_table_writer(*args, **kwargs):  # type: ignore
        _privacy_not_available("Online table writer")


# Proprietary checkpoint download from object storage (omitted in anonymized release)
try:
    from model.tool.loader import download_checkpoint  # type: ignore
except Exception:  # pragma: no cover
    def download_checkpoint(*args, **kwargs):  # type: ignore
        _privacy_not_available("Checkpoint download from object storage")


def DistributedDataset(*args, **kwargs):
    """
    Placeholder for the large-scale distributed dataset wrapper.

    In the internal system, this reads from an internal data platform.
    In this anonymized release, the implementation is omitted; users
    should plug in their own Dataset implementation instead.
    """
    _privacy_not_available("Distributed training dataset")


def DistributedDatasetWithBM25(*args, **kwargs):
    """
    Placeholder for the distributed dataset with local BM25 supervision.

    See `dataset.py` in the original internal project for details; here
    we only preserve the interface for reviewers.
    """
    _privacy_not_available("Distributed training dataset with BM25 supervision")


def train(config):
    # Basic environment logging for distributed training.
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Rank {rank}: start training...")
    print(f"Rank {rank}: distributed env - RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")

    # Call init_distributed_mode() first to set correct CUDA device.
    init_distributed_mode()

    # Now it is safe to check CUDA.
    if torch.cuda.is_available():
        print(f"Rank {rank}: CUDA is available")
        try:
            gpu_count = torch.cuda.device_count()
            print(f"Rank {rank}: detected {gpu_count} GPU(s)")
        except Exception as e:
            print(f"Rank {rank}: warning during GPU count check: {e}")
    else:
        print(f"Rank {rank}: WARNING: CUDA is not available")

    config["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Rank {rank}: config["device"]: {config["device"]}')
    print(
        f"Rank {rank}: distributed init done - "
        f"RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}"
    )

    model = get_model(config)
    random_seed = set_seed_from_config(config)
    optimizer, scheduler = get_optim_scheduler(model, config)
    regularizer = None
    iterations = (1, config["num_training_steps"] + 1)

    resume_path = config.get("resume_path", "")
    if os.path.exists(resume_path):
        print("Resuming training from checkpoint...")
        set_seed(random_seed + 666)
        try:
            print(f"Loading training checkpoint: {resume_path}")
            if torch.cuda.is_available():
                ckpt = torch.load(resume_path, weights_only=False)
            else:
                ckpt = torch.load(resume_path, map_location=config["device"], weights_only=False)
            print("Training checkpoint loaded")
        except Exception as e:
            print(f"Failed to load training checkpoint: {e}")
            print("Retrying with CPU map_location...")
            try:
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                print("Loaded checkpoint on CPU")
            except Exception as e2:
                raise RuntimeError(f"Failed to load training checkpoint: {e2}")

        print(
            "Start from step {}, remain {} iters".format(
                ckpt["step"], config["num_training_steps"] - ckpt["step"]
            )
        )
        iterations = (ckpt["step"] + 1, config["num_training_steps"] + 1)

        restore_model(model, ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "regularizer" in ckpt:
            regularizer = ckpt.get("regularizer", None)

    model.to(config["device"])
    print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), find_unused_parameters=True)
    loss = get_loss(config)
    tokenizer = model.module.tokenizer if hasattr(model, "module") else model.tokenizer
    output_dim = model.module.output_dim if hasattr(model, "module") else model.output_dim
    print('output_dim: ', output_dim)

    if "regularizer" in config and regularizer is None:
        regularizer = {"eval": {"L0": {"loss": init_regularizer("L0")}}, "train": {}}
        for reg in config["regularizer"]:
            temp = {"loss": init_regularizer(config["regularizer"][reg]["reg"]),
                    "targeted_rep": config["regularizer"][reg]["targeted_rep"]}
            d_ = {}
            if "lambda_q" in config["regularizer"][reg]:
                d_["lambda_q"] = RegWeightScheduler(float(config["regularizer"][reg]["lambda_q"]),
                                                    float(config["regularizer"][reg]["T"]))
            if "lambda_d" in config["regularizer"][reg]:
                d_["lambda_d"] = RegWeightScheduler(float(config["regularizer"][reg]["lambda_d"]),
                                                    float(config["regularizer"][reg]["T"]))
            temp["lambdas"] = d_
            regularizer["train"][reg] = temp

    train_datasets = None

    # Optional: BM25 supervision computed locally.
    if config.get("use_bm25_supervision", False):
        print("Using BM25 supervision for training...")
        vocab_size = model.module.output_dim if hasattr(model, "module") else model.output_dim
        bm25_normalize = config.get("bm25_normalize", "l2")
        print(f"BM25 normalization: {bm25_normalize}")
        train_datasets, train_datasets_len = DistributedDatasetWithBM25(
            config["tables"],
            vocab_size,
            tokenizer,
            bm25_normalize
        )
    else:
        train_datasets, train_datasets_len = DistributedDataset(config["tables"])
    config["train_pairs"] = train_datasets_len
    print("train pairs: ", config["train_pairs"])
    train_loader = get_dataloader(mode='train', dataset=train_datasets, config=config, tokenizer=tokenizer, num_workers=1)

    val_loader = None
    if config.get("val_while_training", False):
        raise NotImplementedError  # TODO

    print("+++ BEGIN TRAINING +++")
    trainer = TransformerTrainer(model=model, iterations=iterations, loss=loss, optimizer=optimizer,
                                 config=config, scheduler=scheduler,
                                 regularizer=regularizer,
                                 train_loader=train_loader,
                                 val_loader=val_loader)
    trainer.train()


def infer(config):
    table_writer = None
    if config.get("write_table_while_test", False):
        # In the anonymized release, the actual online write-back to a production
        # data platform is intentionally omitted. The implementation would write
        # inference results (embeddings, scores, etc.) back to an internal table.
        # Users should integrate their own write-back logic here if needed.
        table_writer = None

    # In the internal system, checkpoint paths are resolved via a proprietary
    # object storage service. Here we use a generic local path structure.
    ckpt_dir = os.path.join(
        config.get("checkpoint_base_path", "./checkpoints"),
        config["checkpoint_dir"],
        config["train_ds"],
        config.get("test_ckpt", ""),
    )
    print("ckpt directory: ", ckpt_dir)
    if os.path.exists(ckpt_dir):
        # In the internal system, this downloads from a proprietary object storage.
        # Here we assume checkpoints are already available locally.
        new_ckpt = ckpt_dir
        print("new_ckpt: ", new_ckpt)
        init_distributed_mode()
        config["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('config["device"]: ', config["device"])
        print(
            "RANK: {}, LOCAL_RANK: {}, WORLD_SIZE: {}".format(
                int(os.environ["RANK"]),
                int(os.environ["LOCAL_RANK"]),
                int(os.environ["WORLD_SIZE"]),
            )
        )

        model = get_model(config)
        set_seed_from_config(config)
        regularizer = None

        ckpt_file = os.path.join(new_ckpt, config.get("test_ckpt", "") + ".pth")
        print("Looking for checkpoint file: ", ckpt_file)
        assert os.path.exists(ckpt_file), f"Checkpoint file not found: {ckpt_file}"

        # Safer torch.load with explicit map_location and error handling.
        try:
            print(f"Loading checkpoint file: {ckpt_file}")
            if torch.cuda.is_available():
                # Use LOCAL_RANK environment variable if available.
                local_rank = os.environ.get("LOCAL_RANK", "0")
                device_str = f"cuda:{local_rank}"
                print(f"Using device: {device_str}")
                checkpoint = torch.load(ckpt_file, map_location=device_str, weights_only=False)
            else:
                print("CUDA is not available, loading checkpoint on CPU")
                checkpoint = torch.load(ckpt_file, map_location=config["device"], weights_only=False)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Retrying on CPU...")
            try:
                checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                print("Loaded checkpoint on CPU")
            except Exception as e2:
                raise RuntimeError(f"Failed to load checkpoint on both CUDA and CPU: {e2}")

        config["step"] = checkpoint["step"]
        config["epoch"] = checkpoint["epoch"]
        print("iter: {}, epoch: {}".format(config["step"], config["epoch"]))

        restore_model(model, checkpoint["model_state_dict"])
        if "regularizer" in checkpoint:
            regularizer = checkpoint.get("regularizer", None)

        model.to(config["device"])
        print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), find_unused_parameters=True)
        tokenizer = model.module.tokenizer if hasattr(model, "module") else model.tokenizer
        output_dim = model.module.output_dim if hasattr(model, "module") else model.output_dim
        print('output_dim: ', output_dim)

        del checkpoint
        torch.cuda.empty_cache()

        if regularizer is None:
            regularizer = {"eval": {"L0": {"loss": init_regularizer("L0")}}}

        test_datasets = None
        test_datasets_len = -1

        # Parse `tables` parameter. Support "q_table,d_table" or single table.
        tables_list = config["tables"].split(",")

        if len(tables_list) == 2:
            # Multi-table format: "query_table,doc_table"
            q_table, d_table = tables_list[0].strip(), tables_list[1].strip()
            if config.get("test_q", False) and config.get("test_d", False):
                raise ValueError(
                    "Both test_q and test_d are True. Please enable only one to specify the task type."
                )
            elif config.get("test_q", False):
                selected_table = q_table
                print(f"Selected query table: {selected_table}")
            elif config.get("test_d", False):
                selected_table = d_table
                print(f"Selected document table: {selected_table}")
            else:
                raise ValueError("You must set test_q=True or test_d=True to specify the task type.")
        else:
            # Single-table format.
            selected_table = tables_list[0].strip()
            task_type = "query" if config.get("test_q", False) else "document"
            print(f"Using single table for {task_type} inference: {selected_table}")

        # Build dataset via the distributed dataset wrapper.
        test_datasets, test_datasets_len = DistributedDataset(selected_table)

        print("test pairs: ", test_datasets_len)
        test_loader = get_dataloader(
            mode="test", dataset=test_datasets, config=config, tokenizer=tokenizer, num_workers=5
        )

        print("+++ BEGIN TESTING +++")
        tester = TransformerTester(
            model=model,
            config=config,
            regularizer=regularizer,
            test_loader=test_loader,
            table_writer=table_writer,
        )
        tester.test()
    else:
        print("Can not find ckpt directory!")

    if table_writer is not None:
        table_writer.close()


if __name__ == "__main__":
    # os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET6'

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--train_ds", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--data_source_path", type=str)

    # Data tables path for the (optional) distributed input pipeline.
    # In the anonymized release, the real production data pipeline is intentionally omitted.
    parser.add_argument(
        "--tables",
        type=str,
        default=None,
        help="Input table path(s) for the distributed dataset wrapper.",
    )

    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    merge_args_into_config(config, args)
    config = dict(config)

    # Ensure `tables` is propagated into config.
    if args.tables:
        config["tables"] = args.tables
        print(f"tables override: {args.tables}")

    with open(args.data_source_path) as f:
        train_conf = json.load(f)
        config["data_source"] = train_conf["source"]

    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "infer":
        infer(config)
