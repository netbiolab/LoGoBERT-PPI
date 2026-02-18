import os
import csv
import random
import argparse
import datetime

import numpy as np
import wandb
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from transformers import AutoTokenizer, get_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from torch.optim import AdamW

from logobert.model.LoGo_BERT import LoGo_BERT
from logobert.utils.collate import smart_batching_collate
from logobert.utils.data_load import load_train_objs, load_val_objs


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
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=1800),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank


def _all_gather_variable_length_tensor(t: torch.Tensor):
    assert dist.is_available() and dist.is_initialized()
    device = t.device

    local_len = torch.tensor([t.size(0)], device=device, dtype=torch.long)
    world_size = dist.get_world_size()

    lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lens, local_len)
    lens = torch.stack(lens).view(-1)
    max_len = int(lens.max().item())

    pad_size = max_len - t.size(0)
    if pad_size > 0:
        pad = torch.zeros((pad_size, *t.shape[1:]), device=device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t

    gathered = [torch.zeros_like(t_pad) for _ in range(world_size)]
    dist.all_gather(gathered, t_pad)

    outs = []
    for g, ln in zip(gathered, lens.tolist()):
        outs.append(g[:ln])
    return torch.cat(outs, dim=0)


@torch.no_grad()
def evaluate(model, data_loader, device, has_labels=True):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for batch in data_loader:
        if has_labels:
            input_a, input_b, labels = batch
        else:
            input_a, input_b = batch
            labels = None

        input_a = {k: v.to(device) for k, v in input_a.items()}
        input_b = {k: v.to(device) for k, v in input_b.items()}
        if has_labels:
            labels = labels.to(device)

        with autocast("cuda"):
            outputs = model(input_a, input_b, labels) if has_labels else model(input_a, input_b)
            loss, logits = outputs if has_labels else (None, outputs)

        probs = torch.sigmoid(logits) if has_labels else logits
        all_probs.append(probs.detach())

        if has_labels:
            total_loss += (loss.detach() * labels.size(0)).item()
            all_labels.append(labels.detach())

    local_probs = torch.cat(all_probs, dim=0)
    if has_labels:
        local_labels = torch.cat(all_labels, dim=0)

    if dist.is_available() and dist.is_initialized():
        y_pred = _all_gather_variable_length_tensor(local_probs).cpu().numpy()

        if not has_labels:
            return y_pred

        y_true = _all_gather_variable_length_tensor(local_labels).cpu().numpy()

        tl = torch.tensor([total_loss], device=device, dtype=torch.float32)
        ns = torch.tensor([local_labels.numel()], device=device, dtype=torch.float32)
        dist.all_reduce(tl, op=dist.ReduceOp.SUM)
        dist.all_reduce(ns, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            roc_auc = roc_auc_score(y_true, y_pred)
            aupr = average_precision_score(y_true, y_pred)
            avg_loss = tl.item() / (ns.item() + 1e-12)
            return roc_auc, aupr, avg_loss
        return None, None, None

    y_pred = local_probs.cpu().numpy()
    if not has_labels:
        return y_pred

    y_true = local_labels.cpu().numpy()
    roc_auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    avg_loss = total_loss / (len(y_true) + 1e-12)
    return roc_auc, aupr, avg_loss


def custom_optimizer(model, lr: float, weight_decay: float, verbose: bool = True):
    real_model = model.module if hasattr(model, "module") else model

    decay_params, no_decay_params = [], []
    no_decay_keys = [
        "bias",
        "layernorm", "layer_norm", "ln_", "ln.", "norm", ".ln",
        "sbert_weight", "maxsim_weight",
    ]

    for name, param in real_model.named_parameters():
        if not param.requires_grad:
            continue
        is_1d = (param.ndim == 1)
        name_l = name.lower()
        hit_by_name = any(key in name_l for key in no_decay_keys)

        if is_1d or hit_by_name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
        def nparams(ps):
            return sum(p.numel() for p in ps)
        print(
            f"[Optimizer] decay: {len(decay_params)} tensors, {nparams(decay_params):,} params | "
            f"no-decay: {len(no_decay_params)} tensors, {nparams(no_decay_params):,} params"
        )

    return AdamW(optimizer_grouped_parameters, lr=lr)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epochs,
    local_rank,
    save_path,
    early_stop_patience=3,
    log_path=None,
    start_epoch=0,
    best_aupr=-1.0,
    epochs_no_improve=0,
    grad_accum_steps=1,
    measure_every=100,
    max_grad_norm=1.0,
):
    if log_path and local_rank == 0 and start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "global_step", "train_loss", "roc_auc", "aupr", "val_loss"])

    global_step = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, disable=(local_rank != 0))
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            input_a, input_b, labels = batch
            input_a = {k: v.to(device) for k, v in input_a.items()}
            input_b = {k: v.to(device) for k, v in input_b.items()}
            labels = labels.to(device)

            with autocast("cuda"):
                loss_main, _, extras = model(input_a, input_b, labels, return_extras=True)
            loss = loss_main / grad_accum_steps

            extras["concat"].retain_grad()
            scaler.scale(loss).backward()

            reach_boundary = ((step + 1) % grad_accum_steps == 0)
            if reach_boundary:
                scaler.unscale_(optimizer)

                if (local_rank == 0) and ((global_step + 1) % measure_every == 0):
                    D = extras.get("D", None)
                    if (extras["concat"].grad is not None) and (D is not None):
                        g = extras["concat"].grad.detach()
                        g1 = g[:, : 3 * D]
                        g2 = g[:, 3 * D: 3 * D + 1]
                        g1_mag = g1.abs().mean().item()
                        g2_mag = g2.abs().mean().item()
                        ratio = g2_mag / (g1_mag + 1e-12)
                        wandb.log({
                            "grad/concat_g1_mean_abs": g1_mag,
                            "grad/concat_g2_mean_abs": g2_mag,
                            "grad/g2_over_g1": ratio,
                            "train/global_step": global_step + 1,
                        })
                    else:
                        wandb.log({
                            "dbg/concat_has_grad": int(extras["concat"].grad is not None),
                            "train/global_step": global_step + 1,
                        })

                    mod = model.module if hasattr(model, "module") else model

                    def gnorm(p):
                        return (p.grad.detach().float().norm().item()
                                if (p is not None and p.grad is not None) else 0.0)

                    mw = gnorm(getattr(mod, "maxsim_weight", None))
                    sw = gnorm(getattr(mod, "sbert_weight", None))
                    wandb.log({
                        "grad/maxsim_weight_norm": mw,
                        "grad/sbert_weight_norm": sw,
                        "grad/weight_grad_ratio_mw_over_sw": (mw / (sw + 1e-12)) if sw != 0 else 0.0,
                    })

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if local_rank == 0 and (step % 10 == 0):
                pbar.set_description(f"Epoch {epoch} Step {step} | Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

        result = evaluate(model, val_loader, device, has_labels=True)

        if result is not None and local_rank == 0:
            roc_auc, aupr, val_loss = result
            wandb.log({"val/roc_auc": roc_auc, "val/aupr": aupr, "val/loss": val_loss})
            print(f"Epoch {epoch} | ROC-AUC: {roc_auc:.4f}, AUPR: {aupr:.4f}, Val Loss: {val_loss:.4f}")

            if log_path:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, global_step, loss.item(), roc_auc, aupr, val_loss])

            os.makedirs(save_path, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(save_path, f"epoch{epoch}.pt"))
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--pos_weight", type=float, default=10.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_path", type=str, default="./checkpoints")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--run_name", type=str, default="logobert")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_path", type=str, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--use_ln_g1", action="store_true")
    p.add_argument("--score_norm", type=str, default="none", choices=["none", "tanh"])
    p.add_argument("--hidden_mult", type=int, default=1, choices=[1, 2])
    p.add_argument("--act", type=str, default="relu", choices=["relu", "silu"])
    p.add_argument("--score_fn", type=str, choices=["dot", "cosine"], default="dot")
    p.add_argument("--use_maxsim", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device, local_rank = setup_ddp()

    if local_rank == 0:
        wandb.init(project="logobert-ppi", name=args.run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = load_train_objs(args.train_path)
    val_data = load_val_objs(args.val_path)

    collate_fn = lambda batch: smart_batching_collate(batch, tokenizer, args.max_length, device)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_data),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        sampler=DistributedSampler(val_data, shuffle=False, drop_last=False),
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = LoGo_BERT(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        pos_weight=args.pos_weight,
        use_ln_g1=args.use_ln_g1,
        score_norm=args.score_norm,
        hidden_mult=args.hidden_mult,
        act=args.act,
        score_fn=args.score_fn,
        use_maxsim=args.use_maxsim,
    ).to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )

    optimizer = custom_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    batches_per_epoch = len(train_loader)
    optim_steps_per_epoch = (batches_per_epoch + args.grad_accum_steps - 1) // args.grad_accum_steps
    total_optim_steps = optim_steps_per_epoch * args.epochs
    warmup_steps = int(total_optim_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )
    scaler = GradScaler()

    os.makedirs(args.save_path, exist_ok=True)
    log_path = os.path.join(args.save_path, "train_log.csv")

    start_epoch = 0
    best_aupr = -1.0
    epochs_no_improve = 0

    if args.resume_path:
        ckpt_path = os.path.join(args.resume_path, "best.pt")
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            model.module.load_state_dict(torch.load(ckpt_path, map_location=device))

        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()
            if len(lines) > 1:
                start_epoch = int(lines[-1].split(",")[0]) + 1
                print(f"Resuming from epoch {start_epoch}")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        epochs=args.epochs,
        local_rank=local_rank,
        save_path=args.save_path,
        early_stop_patience=args.early_stop_patience,
        log_path=log_path,
        start_epoch=start_epoch,
        best_aupr=best_aupr,
        epochs_no_improve=epochs_no_improve,
        grad_accum_steps=args.grad_accum_steps,
    )

    if local_rank == 0:
        tokenizer.save_pretrained(args.save_path)
        torch.save(model.module.state_dict(), os.path.join(args.save_path, "final.pt"))
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
