#!/usr/bin/env python3

from __future__ import annotations

import os
import time
import argparse
from datetime import timedelta
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
from Bio import SeqIO

from logobert.model.LoGo_BERT import LoGo_BERT


Embedding = Tuple[torch.Tensor, torch.Tensor]  # (emb, attention_mask)


def now() -> float:
    return time.perf_counter()


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if dist_ready() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist_ready() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_max(x: float, device: torch.device) -> float:
    if dist_ready():
        t = torch.tensor([x], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())
    return float(x)


def setup_ddp(timeout_hours: int = 3) -> Tuple[torch.device, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=timeout_hours),
        )

    return torch.device(f"cuda:{local_rank}"), local_rank


def safe_barrier(local_rank: int):
    if not dist_ready():
        return
    try:
        dev = torch.device(f"cuda:{local_rank}")
        dist.barrier(device_ids=[dev])
    except TypeError:
        dist.barrier()


def load_fasta_dict(fasta_path: str, keep: str = "first") -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        pid = rec.id.split()[0]
        s = str(rec.seq)
        if keep == "first":
            if pid not in seqs:
                seqs[pid] = s
        else:
            seqs[pid] = s
    if is_main_process():
        print(f"[FASTA] loaded {len(seqs)} proteins from {fasta_path}")
    return seqs


def get_unique_ids(df: pd.DataFrame) -> List[str]:
    return sorted(set(df["query"]).union(set(df["text"])))


def iter_length_buckets(
    fasta_subset: Dict[str, str],
    batch_size: int,
    max_length: int,
):
    items = [(pid, seq, min(len(seq), max_length)) for pid, seq in fasta_subset.items()]
    items.sort(key=lambda x: x[2], reverse=True)
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        ids = [pid for pid, _, _ in chunk]
        seqs = [seq for _, seq, _ in chunk]
        used_max_len = chunk[0][2]
        yield ids, seqs, used_max_len


@torch.inference_mode()
def encode_proteins(
    model: LoGo_BERT,
    tokenizer,
    fasta_subset: Dict[str, str],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 64,
    out_dtype: torch.dtype = torch.float32,
    mask_dtype: torch.dtype = torch.bool,
) -> Dict[str, Embedding]:
    model.eval()
    embeddings: Dict[str, Embedding] = {}

    torch.cuda.synchronize(device)

    for ids, seqs, used_max_len in tqdm(
        iter_length_buckets(fasta_subset, batch_size, max_length),
        desc="Embedding proteins",
        disable=(get_rank() != 0),
    ):
        inputs = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=used_max_len,
        )
        input_ids = inputs["input_ids"].to(device, non_blocking=False)
        attention_mask = inputs["attention_mask"].to(device, non_blocking=False)

        emb = model.encode(input_ids, attention_mask)

        for j, pid in enumerate(ids):
            em = emb[j].detach().to("cpu", dtype=out_dtype)
            mk = attention_mask[j].detach().to("cpu", dtype=mask_dtype)
            embeddings[pid] = (em, mk)

        del emb, input_ids, attention_mask, inputs

    return embeddings


def gather_embeddings(
    shard_embeddings: Dict[str, Embedding],
    rank: int,
    world_size: int,
    embedding_save_path: str,
    local_rank: int,
) -> Dict[str, Embedding]:
    shard_path = f"{embedding_save_path}.rank{rank}.pt"
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    torch.save(shard_embeddings, shard_path)

    del shard_embeddings
    torch.cuda.synchronize()

    safe_barrier(local_rank)

    if rank == 0:
        all_embeddings: Dict[str, Embedding] = {}
        for r in range(world_size):
            path = f"{embedding_save_path}.rank{r}.pt"
            part = torch.load(path, map_location="cpu", weights_only=False)
            all_embeddings.update(part)

        torch.save(all_embeddings, embedding_save_path)
        print(f"[rank0] saved {len(all_embeddings)} embeddings → {embedding_save_path}")

        for r in range(world_size):
            os.remove(f"{embedding_save_path}.rank{r}.pt")

    safe_barrier(local_rank)
    return torch.load(embedding_save_path, map_location="cpu", weights_only=False)


@torch.inference_mode()
def infer_pair_scores_batch(
    model: LoGo_BERT,
    df: pd.DataFrame,
    protein_embeddings: Dict[str, Embedding],
    device: torch.device,
    batch_size: int = 1024,
) -> pd.DataFrame:
    model.eval()

    queries = df["query"].tolist()
    texts = df["text"].tolist()
    n = len(df)

    scores_out: List[float] = []

    for i in tqdm(
        range(0, n, batch_size),
        disable=(get_rank() != 0),
        desc=f"Inferring scores (rank={get_rank()})",
    ):
        q_batch = queries[i:i + batch_size]
        t_batch = texts[i:i + batch_size]

        emb_a = [protein_embeddings[q][0] for q in q_batch]
        emb_b = [protein_embeddings[t][0] for t in t_batch]
        mask_a = [protein_embeddings[q][1] for q in q_batch]
        mask_b = [protein_embeddings[t][1] for t in t_batch]

        emb_a_pad = pad_sequence(emb_a, batch_first=True).to(device, non_blocking=True)
        emb_b_pad = pad_sequence(emb_b, batch_first=True).to(device, non_blocking=True)
        mask_a_pad = pad_sequence(mask_a, batch_first=True).to(device, non_blocking=True)
        mask_b_pad = pad_sequence(mask_b, batch_first=True).to(device, non_blocking=True)

        logits = model.predict_from_embeds(
            emb_a_pad, mask_a_pad, emb_b_pad, mask_b_pad,
            return_logits=True,
        )
        probs = torch.sigmoid(logits)
        scores_out.extend(probs.detach().cpu().flatten().tolist())

        del emb_a_pad, emb_b_pad, mask_a_pad, mask_b_pad, logits, probs

    out = df.copy()
    out["score"] = scores_out
    return out


def merge_rank_csvs(output_path: str, world_size: int, poll_seconds: int = 5, timeout_hours: int = 6):
    import csv
    import heapq

    rank_files = [f"{output_path}.rank{r}.csv" for r in range(world_size)]
    deadline = time.time() + timeout_hours * 3600

    while True:
        done = all(os.path.exists(p) for p in rank_files)
        if done or time.time() > deadline:
            break
        time.sleep(poll_seconds)

    if not all(os.path.exists(p) for p in rank_files):
        missing = [p for p in rank_files if not os.path.exists(p)]
        raise RuntimeError(f"[rank0] missing rank output files: {missing}")

    files = []
    readers = []
    heap = []

    for r, path in enumerate(rank_files):
        f = open(path, newline="")
        files.append(f)
        rd = csv.DictReader(f)
        readers.append(rd)
        row = next(rd, None)
        if row is not None:
            heapq.heappush(heap, (int(row["global_idx"]), r, row))

    tmp_final = f"{output_path}.tmp"
    with open(tmp_final, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["query", "text", "score"])
        writer.writeheader()

        last_idx = -1
        written = 0
        while heap:
            gidx, r, row = heapq.heappop(heap)

            if gidx <= last_idx:
                raise RuntimeError(f"[rank0] global_idx not increasing: {gidx} after {last_idx}")
            last_idx = gidx

            writer.writerow({"query": row["query"], "text": row["text"], "score": row["score"]})
            written += 1

            nxt = next(readers[r], None)
            if nxt is not None:
                heapq.heappush(heap, (int(nxt["global_idx"]), r, nxt))

    os.replace(tmp_final, output_path)
    print(f"[rank0] saved merged csv → {output_path} (rows={written})")

    for f in files:
        f.close()
    for rp in rank_files:
        os.remove(rp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair_csv", type=str, required=True, help="CSV with columns: query,text")
    p.add_argument("--hf_repo", type=str, default="hbeen/LoGoBERT-PPI-Eukaryote", help="HF repo id for weights/config")
    p.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Tokenizer source (base PLM)")
    p.add_argument("--fasta_path", type=str, default=None, help="FASTA with protein sequences (required if no cache)")
    p.add_argument("--embeddings_path", type=str, default=None, help="Path to precomputed embeddings (.pt)")
    p.add_argument("--embedding_save_path", type=str, default="protein_embeddings.pt", help="Output path for embedding cache")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--embed_batch_size", type=int, default=64)
    p.add_argument("--inf_batch_size", type=int, default=1024)
    p.add_argument("--output_path", type=str, default="pair_scores.csv")
    p.add_argument("--embed_only", action="store_true", help="Compute/save embeddings then exit")
    p.add_argument("--ddp_timeout_hours", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device, local_rank = setup_ddp(timeout_hours=args.ddp_timeout_hours)

    rank = get_rank()
    world_size = get_world_size()

    df = pd.read_csv(args.pair_csv)
    if not {"query", "text"}.issubset(df.columns):
        raise ValueError("pair_csv must have columns: query, text")

    if is_main_process():
        print(f"[DATA] pairs loaded: {df.shape} from {args.pair_csv}")
        print(f"[DDP] world_size={world_size}")

    unique_ids = get_unique_ids(df)
    chunks = np.array_split(unique_ids, world_size)
    shard_ids = list(chunks[rank])

    if is_main_process():
        print(f"[DATA] total unique IDs: {len(unique_ids)}")
    print(f"[rank {rank}] shard IDs: {len(shard_ids)}")

    use_cache = (args.embeddings_path is not None) and os.path.exists(args.embeddings_path)
    if is_main_process():
        print(f"[CACHE] use cached embeddings: {use_cache}")

    model = LoGo_BERT.from_pretrained(args.hf_repo).to(device)
    model.eval()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    t0 = now()

    if use_cache:
        if is_main_process():
            print(f"[CACHE] loading embeddings: {args.embeddings_path}")
        protein_embeddings = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)

    else:
        if args.fasta_path is None:
            raise ValueError("fasta_path is required when embeddings_path is not provided (no cache).")

        fasta = load_fasta_dict(args.fasta_path)
        fasta_subset = {pid: fasta[pid] for pid in shard_ids if pid in fasta}

        missing_in_fasta = set(shard_ids) - set(fasta_subset.keys())
        if missing_in_fasta:
            print(f"[rank {rank}] warning: {len(missing_in_fasta)} IDs not found in FASTA (ignored)")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        shard_embeddings = encode_proteins(
            model=model,
            tokenizer=tokenizer,
            fasta_subset=fasta_subset,
            device=device,
            max_length=args.max_length,
            batch_size=args.embed_batch_size,
        )

        protein_embeddings = gather_embeddings(
            shard_embeddings=shard_embeddings,
            rank=rank,
            world_size=world_size,
            embedding_save_path=args.embedding_save_path,
            local_rank=local_rank,
        )

        if is_main_process():
            print(f"[CACHE] embeddings saved: {args.embedding_save_path}")

    torch.cuda.synchronize(device)
    t1 = now()

    emb_sec = t1 - t0
    emb_sec_max = ddp_max(emb_sec, device)

    if is_main_process():
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"[TIME] embedding_sec(max over ranks)={emb_sec_max:.3f}")
        print(f"[VRAM] peak_alloc_mb(embedding)={peak_alloc_mb:.2f}")
        print(f"[VRAM] peak_reserved_mb(embedding)={peak_reserved_mb:.2f}")

        missing = set(unique_ids) - set(protein_embeddings.keys())
        if missing:
            print(f"[WARN] {len(missing)} IDs missing in embeddings. Inference will fail if referenced.")

    if args.embed_only:
        safe_barrier(local_rank)
        if dist_ready():
            dist.destroy_process_group()
        return

    shard_df = df.iloc[rank::world_size].copy()
    shard_df["global_idx"] = shard_df.index
    shard_df = shard_df.reset_index(drop=True)

    def _len(pid: str) -> int:
        return int(protein_embeddings[pid][1].sum().item())

    shard_df["_qlen"] = shard_df["query"].map(_len)
    shard_df["_tlen"] = shard_df["text"].map(_len)
    shard_df["_maxlen"] = shard_df[["_qlen", "_tlen"]].max(axis=1)
    shard_df = shard_df.sort_values("_maxlen", ascending=False).reset_index(drop=True)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    t2 = now()
    result_df = infer_pair_scores_batch(
        model=model,
        df=shard_df,
        protein_embeddings=protein_embeddings,
        device=device,
        batch_size=args.inf_batch_size,
    )
    torch.cuda.synchronize(device)
    t3 = now()

    inf_sec = t3 - t2
    inf_sec_max = ddp_max(inf_sec, device)

    if is_main_process():
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        sec_per_pair = inf_sec_max / len(shard_df) if len(shard_df) else float("nan")
        print(f"[TIME] inference_sec(max over ranks)={inf_sec_max:.3f}")
        print(f"[TIME] sec_per_pair(rank0_shard)={sec_per_pair:.6f}")
        print(f"[VRAM] peak_alloc_mb(inference)={peak_alloc_mb:.2f}")
        print(f"[VRAM] peak_reserved_mb(inference)={peak_reserved_mb:.2f}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    rank_out = f"{args.output_path}.rank{rank}.csv"
    tmp_path = f"{rank_out}.tmp"

    result_df = result_df.sort_values("global_idx").reset_index(drop=True)
    result_df = result_df[["query", "text", "global_idx", "score"]]
    result_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, rank_out)

    safe_barrier(local_rank)

    if is_main_process():
        merge_rank_csvs(args.output_path, world_size)

    safe_barrier(local_rank)
    if dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
