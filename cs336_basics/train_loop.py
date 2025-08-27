import os
from datetime import datetime
import time
import torch
import wandb
from cs336_basics.optimizer import gradient_clipping, AdamW, lr_cosine_schedule
from cs336_basics.transformerLM import TransformerLM
from cs336_basics.utils import save_checkpoint,get_batch, cross_entropy, PretrainedConfig
from cs336_basics.experiments import *
from tqdm import tqdm
import numpy.typing as npt
import numpy as np

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
    gradient_clipping(model.parameters(), config.gradient_clipping)

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


def train_model(config: PretrainedConfig):
    global step
    run = wandb.init(
        project=config.project_name,
        name = datetime.now().strftime("%Y%m%d-%H%M%S"),
        config=config.__dict__,
    )
    print("====wandb initialization completed===")

    print(f"torch.cuda.is_available()=={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count()=={torch.cuda.device_count()}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if torch.device(config.device)=='cuda':
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    # 加载数据
    train_data = np.memmap(config.train_path, dtype=np.int32, mode="r")
    valid_data = np.memmap(config.valid_path, dtype=np.int32, mode="r")

    model = TransformerLM(config.vocab_size,
                          config.context_length,
                          config.d_model,
                          config.num_layers,
                          config.num_heads,
                          config.d_ff,
                          theta = config.rope_theta,
                          device = config.device)
    if config.use_compile:
        print("Compiling model for training high performance...")
        # model = torch.compile(model)

    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate,
                      betas=(config.beta1,config.beta2),
                      eps=config.epsilon,
                      weight_decay=config.weight_decay)
    print("train device", torch.device(config.device))
    print("train data size: ", train_data.shape[0], "valid data size: ", valid_data.shape[0])
    total_token_processing = config.batch_size*config.context_length*config.total_steps
    if total_token_processing<327680000:
        print("warning: total_token_processed<327680000, may underfit")
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("trainable model parameters", model.print_trainable_parameters())

    #start training
    start_time = datetime.now()
    start_timestamp = time.time()
    for step in tqdm(range(1, config.total_steps+1)):
        lr = lr_cosine_schedule(
            step,
            config.learning_rate,
            config.learning_rate*0.05,
            config.warmup_steps,
            config.total_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss = train(step, train_data, model, optimizer, config)

        if step%config.log_freq==0:
            grad_norm = torch.sqrt(sum(x * x for x in (p.grad.data.norm() for p in model.parameters() if p.requires_grad)))
            wandb.log({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/wallclock_time": (datetime.now() - start_time).total_seconds(),
                "train/total_steps": step,
            },step=step)
            print(f"step= {step}, loss: {loss}, lr={lr}, grad_norm={grad_norm}")
            logger.info(f"step= {step}, loss: {loss}, lr={lr}, grad_norm={grad_norm}")
        if step%config.eval_freq==0:
            valid_loss = evaluate(valid_data, model, config)
            wandb.log({
                "valid/loss": valid_loss,
                "valid/wallclock_time": time.time() - start_timestamp,
            }, step=step)
            print(f"step= {step}, loss: {valid_loss}")

        # save checkpoint
        if step % config.checkpoint_freq==0:
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pth"))
            print(f"checkpoint has been saved to {os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pth")}")

    # final evaluate loss
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
        os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    )

    wandb.finish()