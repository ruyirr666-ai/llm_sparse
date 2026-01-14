import os
import random
import numpy as np
import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler
from model.transformer_rep import SiameseBase
from model.dataloader import QwDataLoader, TextCollectionDataLoader
from model.tool.losses import InBatchPairwiseNLL, UILoss
from datetime import timedelta


def get_model(config):
    model_map = {
        "SiameseBase": SiameseBase
    }
    

    if "model_class" in config and "init_dict" in config:
        model_class = model_map[config["model_class"]]
        
        return model_class(**config["init_dict"])
    else:
        model_class = SiameseBase
        
        model_args = {
            "model_type_or_dir": config["model_path"],
            "model_type_or_dir_q": config["model_path"] if config.get("seperate_query_model", False) else None,
            "freeze_d_model": config.get("freeze_d_model", False),
            "agg": config.get("agg", "max"),
            "model_type": config.get("model_type", "bert"),
            "torch_dtype": config.get("torch_dtype", "float32"),
            "out_hidden": True,
            "hidden_process": config.get("hidden_process", False),
            "hidden_agg": config.get("hidden_agg", "max"),
            "use_bidirectional_attention": config.get("use_bidirectional_attention", False),
            "use_dual_pass_concat": config.get("use_dual_pass_concat", False)
        }
        
        return model_class(**model_args)


def get_dataloader(mode, dataset, config=None, tokenizer=None,
                   shuffle=False, num_workers=0, sampler=None, drop_last=False):
    batch_size = None
    dataloader_class = None
    if mode == 'train':
        dataloader_map = {
            "qw_pairs": QwDataLoader  # q/d
        }
        batch_size = config["train_batch_size"]
        dataloader_class = dataloader_map[config["train_mode"]]
    elif mode == 'test':
        dataloader_map = {
            "only_one": TextCollectionDataLoader  # t/t_id
        }
        batch_size = config["test_batch_size"]
        dataloader_class = dataloader_map[config["test_mode"]]
    else:
        raise NotImplementedError  # TODO
    max_length = config["max_length"]

    return dataloader_class(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            # pin_memory=True,
                            sampler=sampler,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            config=config,
                            drop_last=drop_last)


def get_loss(config):
    loss = {}
    if config["loss"] == "InBatchPairwiseNLL":
        loss["loss"] = InBatchPairwiseNLL()
        if config.get("d_self_score", False):
            loss["d_self_loss"] = InBatchPairwiseNLL(d_self_score=True)
    else:
        raise NotImplementedError("provide valid loss")
    
    if config.get("use_ui_loss", False):
        ui_threshold = float(config.get("ui_threshold", 1e-6))
        loss["ui_loss"] = UILoss(threshold=ui_threshold)
        print(f"UI Loss enabled, threshold: {ui_threshold}")
    
    if config.get("use_hard_negatives", False):
        print("Hard negative training (integrated in InBatch contrastive loss)")
    
    return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_seed_from_config(config):
    if "random_seed" in config:
        random_seed = config["random_seed"]
    else:
        random_seed = 123
    set_seed(random_seed)
    return random_seed


def get_optim_scheduler(model, config):
    lr = float(config["lr"])
    weight_decay = config["weight_decay"]
    warmup_steps = config["warmup_steps"]
    num_training_steps = config["num_training_steps"]

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler


def restore_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    if len(missing_keys) > 0:
        print("missing_keys: ", missing_keys)
    if len(unexpected_keys) > 0:
        print("unexpected_keys: ", unexpected_keys)
    print("restoring model: ", model.__class__.__name__)


def init_distributed_mode():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    print(f"Rank {rank}: Initializing distributed mode...")
    print(f"Rank {rank}: WORLD_SIZE = {world_size}")
    
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible = os.environ['CUDA_VISIBLE_DEVICES']
        print(f"Rank {rank}: CUDA_VISIBLE_DEVICES = {cuda_visible}")
        
        visible_gpus = [x.strip() for x in cuda_visible.split(',') if x.strip()]
        if len(visible_gpus) == 1:
            LOCAL_RANK = 0
            print(f"Rank {rank}: MDL environment detected, LOCAL_RANK = {LOCAL_RANK}")
        else:
            LOCAL_RANK = int(os.environ.get('LOCAL_RANK', rank % len(visible_gpus)))
            print(f"Rank {rank}: Multi-GPU environment detected, LOCAL_RANK = {LOCAL_RANK}")
    else:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            LOCAL_RANK = int(os.environ.get('LOCAL_RANK', rank % num_gpus))
            print(f"Rank {rank}: Using LOCAL_RANK = {LOCAL_RANK}, total GPU count = {num_gpus}")
        else:
            LOCAL_RANK = 0
            print(f"Rank {rank}: CUDA not available, LOCAL_RANK = {LOCAL_RANK}")

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(LOCAL_RANK)
            print(f"Rank {rank}: Successfully set CUDA device to {LOCAL_RANK}")
            
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_count = torch.cuda.device_count()
            print(f"Rank {rank}: Current CUDA device: {current_device}, device name: {device_name}, total device count: {device_count}")
            
            test_tensor = torch.tensor([1.0], device=f'cuda:{LOCAL_RANK}')
            print(f"Rank {rank}: CUDA test tensor created successfully: {test_tensor.device}")
            
        except Exception as e:
            print(f"Rank {rank}: Failed to set CUDA device: {e}")
            print(f"Rank {rank}: Available CUDA device count: {torch.cuda.device_count()}")
            raise
    
    if not torch.distributed.is_initialized():
        print(f"Rank {rank}: Initializing distributed process group...")
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            backend = "nccl"
            print(f"Rank {rank}: Using NCCL backend")
        else:
            backend = "gloo"
            print(f"Rank {rank}: CUDA not available or no GPU, using Gloo backend")
        
        try:
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=18000)
            )
            print(f"Rank {rank}: Distributed process group initialized, backend: {backend}")
        except Exception as e:
            print(f"Rank {rank}: {backend} backend initialization failed: {e}")
            if backend == "nccl":
                print(f"Rank {rank}: Trying to use Gloo backend as backup...")
                try:
                    torch.distributed.init_process_group(
                        backend="gloo",
                        world_size=world_size,
                        rank=rank,
                        timeout=timedelta(seconds=18000)
                    )
                    print(f"Rank {rank}: Gloo backend initialization successful")
                except Exception as e2:
                    print(f"Rank {rank}: Gloo backend also failed: {e2}")
                    raise
            else:
                raise
    else:
        print(f"Rank {rank}: Distributed process group already initialized")


def merge_args_into_config(config, args):
    kvs = args._get_kwargs()
    for k, v in kvs:
        if k in config:
            print('Overwrite: k {}, old v {}, new v {}'.format(k, config[k], v))
        config[k] = v
