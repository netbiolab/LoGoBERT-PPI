import os, csv, random, argparse
import csv
import random
import numpy as np
import wandb
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from logobert.model.baseline import ESM2_MLP
from logobert.utils.collate import smart_batching_collate
from logobert.utils.data_load import load_train_objs, load_val_objs
from torch.optim import AdamW

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank

@torch.no_grad()
def evaluate(model, data_loader, device, pos_weight=10.0, has_labels=True):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    if has_labels:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    for batch in data_loader:
        input_a, input_b, labels = batch if has_labels else (*batch, None)
        input_a = {k: v.to(device) for k, v in input_a.items()}
        input_b = {k: v.to(device) for k, v in input_b.items()}
        if has_labels:
            labels = labels.to(device)

        with autocast("cuda"):
            outputs = model(input_a, input_b, labels
                            #, use_sbert_concat=True
                            ) if has_labels else model(input_a, input_b)
            loss, logits = outputs if has_labels else (None, outputs)

        probs = torch.sigmoid(logits) if has_labels else logits
        all_logits.append(probs.detach())
        if has_labels:
            total_loss += loss.item()
            all_labels.append(labels.detach())

    local_logits = torch.cat(all_logits)
    if has_labels:
        local_labels = torch.cat(all_labels)

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_logits = [torch.zeros_like(local_logits) for _ in range(world_size)]
        dist.all_gather(gathered_logits, local_logits)
        y_pred = torch.cat(gathered_logits).cpu().numpy()

        if has_labels:
            gathered_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
            dist.all_gather(gathered_labels, local_labels)
            y_true = torch.cat(gathered_labels).cpu().numpy()

            if dist.get_rank() == 0:
                roc_auc = roc_auc_score(y_true, y_pred)
                aupr = average_precision_score(y_true, y_pred)
                avg_loss = total_loss / len(data_loader)
                return roc_auc, aupr, avg_loss
            else:
                return None, None, None
        else:
            return y_pred
    else:
        y_pred = local_logits.cpu().numpy()
        if has_labels:
            y_true = local_labels.cpu().numpy()
            roc_auc = roc_auc_score(y_true, y_pred)
            aupr = average_precision_score(y_true, y_pred)
            avg_loss = total_loss / len(data_loader)
            return roc_auc, aupr, avg_loss
        else:
            return y_pred

def train(model, train_loader, val_loader, optimizer, scheduler, scaler, device,
          epochs, local_rank, save_path, pos_weight=10.0,
          early_stop_patience=3, log_path=None, start_epoch=0,
          best_aupr=-1.0, epochs_no_improve=0, grad_accum_steps=1):
    if log_path and local_rank == 0 and start_epoch == 0:
        with open(log_path, "w") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "roc_auc", "aupr", "val_loss"])

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, disable=(local_rank != 0))

        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_a, input_b, labels = batch
            input_a = {k: v.to(device) for k, v in input_a.items()}
            input_b = {k: v.to(device) for k, v in input_b.items()}
            labels = labels.to(device)

            with autocast("cuda"):
                loss, _ = model(input_a, input_b, labels
                                #, use_sbert_concat=True
                                )

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1 == len(pbar)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if local_rank == 0 and step % 10 == 0:
                pbar.set_description(f"Epoch {epoch} Step {step} | Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

        result = evaluate(model, val_loader, device, pos_weight)
        if result is not None and local_rank == 0:
            roc_auc, aupr, val_loss = result
            wandb.log({"val/roc_auc": roc_auc, "val/aupr": aupr, "val/loss": val_loss})
            print(f"Epoch {epoch} | ROC-AUC: {roc_auc:.4f}, AUPR: {aupr:.4f}, Val Loss: {val_loss:.4f}")

            if log_path:
                with open(log_path, "a") as f:
                    csv.writer(f).writerow([epoch, step, loss.item(), roc_auc, aupr, val_loss])

            torch.save(model.module.state_dict(), os.path.join(save_path, f"epoch{epoch}.pt"))
            torch.save(model.module.state_dict(), os.path.join(save_path, "best.pt"))
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            torch.save(scaler.state_dict(), os.path.join(save_path, "scaler.pt"))

            if aupr > best_aupr:
                best_aupr = aupr
                epochs_no_improve = 0
                torch.save(model.module.state_dict(), os.path.join(save_path, "best.pt"))
                print("New best model saved.")
            else:
                epochs_no_improve += 1
                print(f"No improvement in AUPR for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

def main(args):
    set_seed(args.seed)
    device, local_rank = setup_ddp()
    if local_rank == 0:
        wandb.init(project="baseline", name=args.run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data = load_train_objs(args.train_path)
    val_data = load_val_objs(args.val_path)
    collate_fn = lambda batch: smart_batching_collate(batch, tokenizer, args.max_length, device)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              sampler=DistributedSampler(train_data),
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            sampler=DistributedSampler(val_data),
                            collate_fn=collate_fn, drop_last=True)

    model = ESM2_MLP(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        pos_weight=args.pos_weight,
        #use_cpp_maxsim=args.use_cpp_maxsim
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    os.makedirs(args.save_path, exist_ok=True)
    log_path = os.path.join(args.save_path, "train_log.csv")
    start_epoch = 0
    best_aupr = -1.0
    epochs_no_improve = 0

    if args.resume_path:
        print(f"Resuming from {args.resume_path}")
        model.module.load_state_dict(torch.load(os.path.join(args.resume_path, "best.pt"), map_location=device))
        if os.path.exists(os.path.join(args.resume_path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(args.resume_path, "optimizer.pt"), map_location=device))
        if os.path.exists(os.path.join(args.resume_path, "scheduler.pt")):
            scheduler.load_state_dict(torch.load(os.path.join(args.resume_path, "scheduler.pt"), map_location=device))
        if os.path.exists(os.path.join(args.resume_path, "scaler.pt")):
            scaler.load_state_dict(torch.load(os.path.join(args.resume_path, "scaler.pt")))
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    start_epoch = int(lines[-1].split(',')[0]) + 1
                    print(f"Resuming from epoch {start_epoch}")

    train(model, train_loader, val_loader, optimizer, scheduler, scaler, device,
          args.epochs, local_rank, args.save_path, args.pos_weight,
          args.early_stop_patience, log_path, start_epoch, best_aupr, epochs_no_improve, args.grad_accum_steps)

    if local_rank == 0:
        tokenizer.save_pretrained(args.save_path)
        torch.save(model.module.state_dict(), os.path.join(args.save_path, "final.pt"))
        print(f"Final model and tokenizer saved at {args.save_path}")
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--pos_weight", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    args = parser.parse_args()
    main(args)
    
    

