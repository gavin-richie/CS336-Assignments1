import os
from datetime import datetime
import time
from typing import Dict

import torch
import wandb
from cs336_basics.optimizer import gradient_clipping, AdamW, lr_cosine_schedule
from cs336_basics.transformerLM import TransformerLM
from cs336_basics.utils import save_checkpoint,get_batch, cross_entropy, PretrainedConfig
from cs336_basics.experiments import *
from tqdm import tqdm
import numpy.typing as npt
import numpy as np
from torch.amp import autocast

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

def train(step, dataset: npt.NDArray, model: torch.nn.Module, optimizer: torch.optim.Optimizer,config):
    inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)

    model.train()

    logits = model(inputs)

    loss = cross_entropy(logits, targets)
    # 优化器梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 梯度裁剪
    gradient_clipping(model.parameters(), config.clip_grad_norm)

    # 梯度更新
    optimizer.step()

    return loss.item()


def evaluate(dataset, model: torch.nn.Module, config):
    model.eval()
    loss_arr = []
    with torch.no_grad():
        inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss_arr.append(loss.item())

    loss_avg = np.mean(loss_arr)
    return loss_avg


def train_model(config: Dict,model_config: Dict):
    global step
    run = wandb.init(
        project=config.project_name,
        name = datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=config.__dict__,
    )
    print("====wandb initialization completed===")
    print("vocab_size=", model_config.vocab_size)
    print(f"torch.cuda.is_available()=={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count()=={torch.cuda.device_count()}")

    os.makedirs(config.ckpt_path, exist_ok=True)

    if torch.device(config.device).type=='cuda':
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    if torch.cuda.is_available():
        if model_config.dtype == "bfloat16":
            max_flops = 989e12  # BF16 Tensor Core without sparsity.
        else:
            max_flops = 989e12 / 2  # TF32 Tensor Core without sparsity.
    else:
        max_flops = float("inf")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if model_config.dtype == "bfloat16" else torch.float32

    # 加载数据
    # train_data = np.memmap(config.train_data_path, dtype=np.int32, mode="r")
    # valid_data = np.memmap(config.valid_data_path, dtype=np.int32, mode="r")
    train_data = np.load(config.train_npy_path, mmap_mode="r")
    valid_data = np.load(config.valid_npy_path, mmap_mode="r")
    # train_data = np.memmap(config.train_npy_path, dtype=np.uint16, mode='r')
    # valid_data = np.memmap(config.valid_npy_path, dtype=np.uint16, mode='r')

    model = TransformerLM(model_config.vocab_size,
                          model_config.context_length,
                          model_config.d_model,
                          model_config.num_layers,
                          model_config.num_heads,
                          model_config.d_ff,
                          theta = model_config.rope_theta,
                          device = device,
                          dtype=dtype)
    if config.use_compile:
        print("Compiling model for training high performance...")
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(),
                      lr=config.lr,
                      betas=(config.beta1,config.beta2),
                      eps=config.epsilon,
                      weight_decay=config.weight_decay)
    print("train device", torch.device(model_config.device))
    print("train data size: ", train_data.shape[0], "valid data size: ", valid_data.shape[0])
    total_token_processing = config.batch_size*config.context_length*config.train_steps
    if total_token_processing<327680000:
        print("warning: total_token_processed<327680000, may underfit")
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("trainable model parameters", model.print_trainable_parameters())

    #start training
    start_time = datetime.now()
    start_timestamp = time.time()
    for step in tqdm(range(1, config.train_steps+1)):
        t0 = time.time()
        lr = lr_cosine_schedule(
            step,
            config.min_lr,
            config.lr,
            config.warmup_iters,
            config.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast(device.type, dtype=dtype):
            loss = train(step, train_data, model, optimizer, config)

        if step%config.log_interval==0:
            grad_norm = torch.sqrt(sum(x * x for x in (p.grad.data.norm() for p in model.parameters() if p.requires_grad)))
            wandb.log({
                "train/loss": loss,
                "train/perplexity": torch.exp(torch.tensor(loss)).item(),
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/wallclock_time": (datetime.now() - start_time).total_seconds(),
                "train/total_steps": step,
                "train/gpu_memory": torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0,
            },step=step)
            # wait on the CPU for all device work to end so we get accurate per-iteration timings below
            if device == torch.device("mps"):
                torch.mps.synchronize()
            elif device == torch.device("cuda"):
                torch.cuda.synchronize()
            # time and print
            t1 = time.time()
            # the 0th iteration is often an outlier (much slower) => skip logging it
            token_per_second = step * config.batch_size * config.context_length / (t1-t0)
            print(f"step= {step}|loss={loss:.3f}|lr={lr:.5f}|grad_norm={grad_norm:.5f}|time={t1-t0:.3f}ms|tok/s={token_per_second:.1f}")
            logger.info(f"step= {step}|loss={loss:.3f}|lr={lr:.3f}|grad_norm={grad_norm:.3f}|time={t1-t0:.3f}")
        if step%config.eval_interval==0:
            with autocast(device.type, dtype=dtype):
                valid_loss = evaluate(valid_data, model, config)
            wandb.log({
                "valid/loss": valid_loss,
                "valid/perplexity": torch.exp(torch.tensor(valid_loss)).item(),
                "valid/wallclock_time": time.time() - start_timestamp,
            }, step=step)
            print(f"step= {step}, eval_loss: {valid_loss}")

        # save checkpoint
        if step % config.checkpoint_freq==0:
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(config.ckpt_path, f"checkpoint_{step}.pth"))
            print(f"checkpoint has been saved to {os.path.join(config.ckpt_path, f"checkpoint_{step}.pth")}")

    # final evaluate loss
    with autocast(device.type, dtype=dtype):
        eval_loss = evaluate(valid_data, model, config)
    wandb.log({
        'val/loss': eval_loss,
        'val/wallclock_time': time.time() - start_timestamp
    }, step=step)
    print(f"final evaluation loss: {eval_loss}")

    save_checkpoint(
        model,
        optimizer,
        step,
        os.path.join(config.ckpt_path, f"checkpoint_{step}.pt")
    )

    wandb.finish()