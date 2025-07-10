import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Adafactor, get_linear_schedule_with_warmup, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm, trange
from datetime import datetime
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning, message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")


class ParallelSet(Dataset):
    def __init__(self, dataframe, source, target):
        self.df = dataframe
        self.source = source
        self.target = target

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.loc[idx, self.source], self.df.loc[idx, self.target]

def collate_fn(examples, tokenizer, source, target):
    max_length = 256
    src = [line[0] for line in examples]
    dst = [line[1] for line in examples]
    tokenizer.source = source
    inputs = tokenizer(text=src, return_tensors='pt', padding="max_length", truncation=True, max_length=max_length)
    tokenizer.source = target
    references = tokenizer(text=dst, return_tensors='pt', padding="max_length", truncation=True, max_length=max_length)
    references.input_ids[references.input_ids == tokenizer.pad_token_id] = -100
    inputs["labels"] = references.input_ids
    return inputs

def train(model, optimizer, scheduler, device, train_dl, valid_dl, n_epoch, log_dir, save_interval):
    writer = SummaryWriter(log_dir)
    best_loss = float("inf")
    step = 0

    for epoch in trange(n_epoch, desc="Epoch"):
        model.train()
        total_loss = 0
        batch_count = 0
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch+1} Training") as pbar:  # 创建一个进度条
            for batch_num, batch in enumerate(train_dl):
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss.mean()  # Take the mean of the loss across all GPUs
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                batch_count += 1
                pbar.update(1)  # 更新进度条
                pbar.set_description(f"Epoch {epoch+1} Training - Batch {batch_num+1}, Loss: {loss.item():.4f}")  # 更新描述信息

                if step % 500 == 0:
                    average_loss = total_loss / batch_count
                    writer.add_scalar("Loss/Training", average_loss, step)
                    model.eval()
                    valid_loss = []
                    for valid_batch in tqdm(valid_dl, desc="Validation"):
                        valid_batch = {k: v.to(device) for k, v in valid_batch.items()}
                        v_loss = model(**valid_batch).loss
                        v_loss = v_loss.mean()  # Also take the mean for validation loss
                        valid_loss.append(v_loss.item())
                    ave_valid_loss = np.mean(valid_loss)
                    writer.add_scalar("Loss/Validation", ave_valid_loss, step)
                    if ave_valid_loss < best_loss:
                        best_loss = ave_valid_loss
                        save_dir = f"{log_dir}/checkpoint_{step}"
                        os.makedirs(save_dir, exist_ok=True)
                        model.module.save_pretrained(save_dir)  # Use .module to save the original model without DataParallel wrapper
                        tokenizer.save_pretrained(save_dir)
                    # Reset the loss for the next 500 steps
                    total_loss = 0
                    batch_count = 0
                step += 1

    save_dir = f"{log_dir}/final"
    os.makedirs(save_dir, exist_ok=True)
    model.module.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'  # 设置使用 GPU 0 和 GPU 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('data/50k_example.csv')
    rand_idx = np.random.permutation(len(df))
    valid_size = 300
    train_idx = rand_idx[:-valid_size]
    valid_idx = rand_idx[-valid_size:]

    source, target = "input", "output"
    model_id = 'model/t5-large'
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    trainset = ParallelSet(df.iloc[train_idx].reset_index(drop=True), source, target)
    validset = ParallelSet(df.iloc[valid_idx].reset_index(drop=True), source, target)

    train_dl = DataLoader(trainset, batch_size=30, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer, source, target))
    valid_dl = DataLoader(validset, batch_size=30, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer, source, target))

    model = T5ForConditionalGeneration.from_pretrained(model_id)
    model = torch.nn.DataParallel(model)  # 包装模型以使用多 GPU
    model.to(device)

    optimizer = Adafactor(model.parameters(), lr=1e-5, relative_step=False, warmup_init=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=1000 * len(train_dl))

    log_dir = f"log/{datetime.now().strftime('%Y_%m_%d-%H_%M')}"
    train(model, optimizer, scheduler, device, train_dl, valid_dl, n_epoch=50, log_dir=log_dir, save_interval=100)
