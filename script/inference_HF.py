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

def ddp_max(x: float, device):
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([x], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())
    return float(x)

def now():
    return time.perf_counter()

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0 

def get_world_size():
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

def setup_ddp(): 
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) 
    torch.cuda.set_device(local_rank) 
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://', 
            rank=int(os.environ.get("RANK", 0)), 
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            timeout=timedelta(hours=3)
        )
    return torch.device(f"cuda:{local_rank}"), local_rank
    
def safe_barrier(local_rank):
    try:
        dev = torch.device(f"cuda:{local_rank}")
        dist.barrier(device_ids=[dev])  
    except TypeError:
        dist.barrier()

def load_fasta_dict(fasta_path, keep='first'):
    d = {}
    for rec in SeqIO.parse(fasta_path, 'fasta'):
        pid = rec.id.split()[0]
        seq = str(rec.seq)
        if keep == 'first':
            if pid not in d:
                d[pid] = seq
        else:
            d[pid] = seq
    print(f'Load {len(d)} protein sequences from {fasta_path}')
    return d
    
    
def get_unique_ids(df):
    return sorted(set(df['query']).union(set(df['text'])))

def _length_buckets(fasta_subset, batch_size, max_length): 
    items = [(pid, seq, min(len(seq), max_length)) for pid, seq in fasta_subset.items()]#.items()
    items.sort(key=lambda x: x[2], reverse=True)
    for i in range(0, len(items), batch_size):
        chunk = items[i:i+batch_size]
        ids = [pid for pid, _, _ in chunk]
        seqs = [seq for _, seq, _ in chunk]
        used_max_len = chunk[0][2]
        yield ids, seqs, used_max_len 


        
@torch.inference_mode()
def encode_proteins(model,
                    tokenizer,
                    fasta_subset,
                    device,
                    max_length=512,
                    batch_size=64,
                    #save_half=True
                    ):
    embeddings = {}
    model.eval()
 
    torch.cuda.synchronize(device)
    for ids, seqs, used_max_len in tqdm(
        _length_buckets(fasta_subset, batch_size, max_length),
        desc='Embedding proteins', disable=(get_rank()!=0)):

        inputs = tokenizer(
            seqs, return_tensors='pt',
            padding=True, truncation=True, max_length=used_max_len)

        input_ids = inputs['input_ids'].to(device, non_blocking=False)
        attention_mask = inputs['attention_mask'].to(device, non_blocking=False)

        emb = model.encode(input_ids, attention_mask)
            
        out_dtype = torch.float32#torch.float16 if save_half else torch.float32
        mask_dtype = torch.bool
        
        
        for j, pid in enumerate(ids):
            em = emb[j].detach().to('cpu', dtype=out_dtype)
            mk = attention_mask[j].detach().to('cpu', dtype=mask_dtype)
            embeddings[pid] = (em, mk)#embeddings[pid] = {'embedding': em, 'attention_mask': mk, 'length': int(mk.sum().item())}
        
        del emb, input_ids, attention_mask, inputs

    return embeddings



def gather_embeddings(shard_embeddings, rank, world_size, embedding_save_path, local_rank):
    shard_path = f"{embedding_save_path}.rank{rank}.pt"
    d = os.path.dirname(shard_path)
    if d:  
        os.makedirs(d, exist_ok=True)

    torch.save(shard_embeddings, shard_path)


    del shard_embeddings
    torch.cuda.synchronize()

    safe_barrier(local_rank)

    if rank == 0:
        all_embeddings = {}
        for r in range(world_size):
            path = f"{embedding_save_path}.rank{r}.pt"
            part = torch.load(path, map_location='cpu', weights_only=False)
            all_embeddings.update(part)

        torch.save(all_embeddings, embedding_save_path)
        print(f"[Rank 0] Total {len(all_embeddings)} embeddings saved to {embedding_save_path}")
        for r in range(world_size):
            os.remove(f"{embedding_save_path}.rank{r}.pt")

    safe_barrier(local_rank)
    return torch.load(embedding_save_path, map_location='cpu', weights_only=False)

@torch.inference_mode()
def infer_pair_scores_batch(model, df, protein_embeddings, device, batch_size=16
                           ):
    model.eval()
    #logits_out, scores_out = [], []
    scores_out = []
    
    
    rank = get_rank()

    
    
    queries = df['query'].tolist()
    texts   = df['text'].tolist()
    n = len(df)

    for i in tqdm(range(0, n, batch_size),
                  disable=(get_rank() != 0),
                  desc=f'Inferring scores (rank = {rank})'):
        q_batch = queries[i:i+batch_size]
        t_batch = texts[i:i+batch_size]

        emb_a  = [protein_embeddings[q][0] for q in q_batch]
        emb_b  = [protein_embeddings[t][0] for t in t_batch]
        mask_a = [protein_embeddings[q][1] for q in q_batch]
        mask_b = [protein_embeddings[t][1] for t in t_batch]

        emb_a_pad  = pad_sequence(emb_a,  batch_first=True).to(device, non_blocking=True)
        emb_b_pad  = pad_sequence(emb_b,  batch_first=True).to(device, non_blocking=True)
        mask_a_pad = pad_sequence(mask_a, batch_first=True).to(device, non_blocking=True)
        mask_b_pad = pad_sequence(mask_b, batch_first=True).to(device, non_blocking=True)
 
        probs = model.predict_from_embeds(
            emb_a_pad, mask_a_pad, emb_b_pad, mask_b_pad
            )

        scores_out.extend(probs.detach().cpu().tolist())
    
        del emb_a_pad, emb_b_pad, mask_a_pad, mask_b_pad, probs
        
    df['score'] = scores_out
    return df

def main(args):


    device, local_rank = setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    
    df_ids = pd.read_csv(args.pair_csv)
    
    assert {'query','text'}.issubset(df_ids.columns), \
        "pair_csv must have columns: query, text"

    if is_main_process():
        print(f"(rank {rank}) Pair file loaded: {df_ids.shape}") 
    
    
    unique_ids = get_unique_ids(df_ids)
    chunks = np.array_split(unique_ids, world_size)
    shard_ids = list(chunks[rank])
    
    if is_main_process():
        print(f"World size: {world_size}, Total unique IDs: {len(unique_ids)}")
    print(f"[Rank {rank}] My shard count: {len(shard_ids)}")

    use_cache = (args.embeddings_path is not None) and os.path.exists(args.embeddings_path)
    if is_main_process():
        print(f"[INFO] Use cached embeddings: {use_cache}")

    model = LoGo_BERT.from_pretrained(args.hf_repo).to(device)
    model.eval()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    t_emb0 = now()

    if use_cache:

        if is_main_process():
            print(f"[INFO] Loading cached embeddings from {args.embeddings_path}")
        protein_embeddings = torch.load(args.embeddings_path, map_location='cpu', weights_only=False)

        missing = set(unique_ids) - set(protein_embeddings.keys())
        if is_main_process() and missing:
            print(f"[WARN] {len(missing)} IDs missing in cached embeddings. They will cause KeyError if referenced.")
    else:
        assert args.fasta_path is not None, "No embeddings_path provided and fasta_path is None."
        fasta_dict = load_fasta_dict(args.fasta_path)
        
        fasta_subset = {pid: fasta_dict[pid] for pid in shard_ids if pid in fasta_dict}
        missing_in_fasta = set(shard_ids) - set(fasta_subset.keys())
        if missing_in_fasta:
            print(f"[Rank {rank}] WARNING: {len(missing_in_fasta)} IDs not in FASTA (ignored for embedding)")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        shard_embeddings = encode_proteins(
            model, tokenizer, fasta_subset, device,
            args.max_length, args.batch_size
        )
        
        protein_embeddings = gather_embeddings(
            shard_embeddings, rank, world_size, args.embedding_save_path, local_rank
        )

        
        if is_main_process():
            print(f"[INFO] Embeddings saved to {args.embedding_save_path}")

    torch.cuda.synchronize(device)
    t_emb1 = now()
    emb_sec = t_emb1 - t_emb0
    emb_sec_max = ddp_max(emb_sec, device)
    if is_main_process():
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        print(f"[TIME] EMBEDDING_SEC(max over ranks)={emb_sec_max:.3f}")
        print(f"PEAK_VRAM_ALLOC_MB_EMB={peak_alloc_mb:.2f}")
        print(f"PEAK_VRAM_RESERVED_MB_EMB={peak_reserved_mb:.2f}")
        print(f"Total gathered/loaded embeddings: {len(protein_embeddings)} (should be ≥ used IDs)")
        missing = set(unique_ids) - set(protein_embeddings.keys())
        if missing:
            print(f"[WARN] {len(missing)} IDs missing in embeddings. Inference may fail if they appear in pairs.")
    if args.embed_only:
        if is_main_process():
            print("[INFO] embed_only=True ")
        dist.barrier()
        dist.destroy_process_group()
        return



    #step4: inference

    shard_df = df_ids.iloc[rank::world_size].copy()
    shard_df['global_idx'] = shard_df.index
    shard_df = shard_df.reset_index(drop=True)
    
    def _len(pid):
        return int(protein_embeddings[pid][1].sum().item())
    
    shard_df['_qlen'] = shard_df['query'].map(_len)
    shard_df['_tlen'] = shard_df['text'].map(_len)
    shard_df['_maxlen'] = shard_df[['_qlen', '_tlen']].max(axis = 1)
    shard_df = shard_df.sort_values('_maxlen', ascending=False).reset_index(drop = True)
    
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t_inf0 = now()
    
    result_df = infer_pair_scores_batch(
        model, shard_df,
        protein_embeddings, device,
        batch_size=args.inf_batch_size
    )
    
    torch.cuda.synchronize(device)
    t_inf1 = now()

    inf_sec = t_inf1 - t_inf0
    inf_sec_max = ddp_max(inf_sec, device)
    if is_main_process():
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        sec_per_pair = inf_sec_max / len(shard_df) if len(shard_df) else float("nan")
        print(f"[TIME] INFERENCE_SEC(max over ranks)={inf_sec_max:.3f}")
        print(f"[TIME] SEC_PER_PAIR(rank0_shard)={sec_per_pair:.6f}")
        print(f"PEAK_VRAM_ALLOC_MB_INF={peak_alloc_mb:.2f}")
        print(f"PEAK_VRAM_RESERVED_MB_INF={peak_reserved_mb:.2f}")
    

    
    #step5: 

    dir_name = os.path.dirname(args.output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    rank_output_path = f"{args.output_path}.rank{rank}.csv"
    tmp_path = f"{rank_output_path}.tmp"
    result_df = result_df.sort_values('global_idx').reset_index(drop = True)
    
    result_df = result_df[['query','text','global_idx','score']]
    

    result_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, rank_output_path)
    

    if is_main_process():
        import csv, heapq


        rank_files = [f"{args.output_path}.rank{r}.csv" for r in range(world_size)]
        deadline = time.time() + 6*3600        
        while True:
            done = all(os.path.exists(p) for p in rank_files)
            if done or time.time() > deadline:
                break
            time.sleep(5)

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

        tmp_final = f"{args.output_path}.tmp"
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

        os.replace(tmp_final, args.output_path)
        print(f"[rank 0] saved final merged csv → {args.output_path} (rows={written})")

        for f in files:
            f.close()

        for rp in rank_files:
            os.remove(rp)



    dist.barrier()
    dist.destroy_process_group()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_csv", type=str, required=True, help="Path to CSV file with columns: query,text")
    parser.add_argument("--hf_repo", type=str, default="hbeen/LoGoBERT-PPI-Eukaryote", help="HF repo id for weights/config")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--embedding_save_path", type=str, default="protein_embeddings.pt")
    parser.add_argument("--output_path", type=str, default="pair_scores.csv")
    parser.add_argument("--inf_batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--fasta_path", type=str, required=False,  
                        help="FASTA with protein IDs used in pair_csv (q_id,t_id)")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Use precomputed embeddings (.pt) if provided")
    parser.add_argument("--embed_only", action="store_true",
                        help="If set, compute/save embeddings and exit without pair inference.")
    args = parser.parse_args()

    main(args)
