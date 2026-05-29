#!/usr/bin/env python3
import os, sys, argparse, glob, torch, numpy as np
from typing import List, Tuple, Dict

def read_fasta(fp: str) -> List[Tuple[str, str]]:
    items = []
    header, seq = None, []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    items.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header is not None:
        items.append((header, "".join(seq)))
    return items

def parse_chain_id(header: str) -> str:

    parts = header.split("|")
    for p in parts:
        p = p.strip()
        if p.lower().startswith("chain_") and len(p) >= 7:
            return p.split("_", 1)[1]
    # 兜底
    return "UNK"

def system_name_from_path(fasta_path: str) -> str:
    return os.path.basename(os.path.dirname(fasta_path))

def collect_fastas(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.fasta"), recursive=True))

@torch.no_grad()
def esm_embed_sequence(seq: str, model, alphabet, device, repr_layer: int, max_tokens: int = 1022):

    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    def _embed_chunk(subseq: str):
        data = [("seq", subseq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]  # [1, T, d]
        reps = reps[0, 1:1+len(subseq), :]  # [L, d]
        return reps.detach().cpu()

    L = len(seq)
    if L <= max_tokens:
        per_res = _embed_chunk(seq)                            # [L,d]
        mean = per_res.mean(dim=0)                             # [d]
        return per_res.float().numpy(), mean.float().numpy()

    chunks = []
    for start in range(0, L, max_tokens):
        sub = seq[start: start + max_tokens]
        chunks.append(_embed_chunk(sub))                      
    per_res = torch.cat(chunks, dim=0)[:L]                     # [L,d]
    mean = per_res.mean(dim=0)
    return per_res.float().numpy(), mean.float().numpy()

def main():
    ap = argparse.ArgumentParser(description="Convert FASTA chains to ESM embeddings with mirrored directory structure.")
    ap.add_argument("--src_root", default=r"G:\TriComplex\database\dataset_3359\ProteinSeq")
    ap.add_argument("--out",default=r"G:\TriComplex\database\dataset_3359\ESM-3B")
    ap.add_argument("--model", default="esm2_t36_3B_UR50D",
                    choices=["esm2_t6_8M_UR50D","esm2_t12_35M_UR50D","esm2_t30_150M_UR50D","esm2_t33_650M_UR50D","esm2_t36_3B_UR50D","esm2_t48_15B_UR50D"])
    ap.add_argument("--mean-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    
    import esm
    load_fn = getattr(esm.pretrained, args.model)
    model, alphabet = load_fn()
    try:
        repr_layer = model.num_layers 
    except Exception:
        repr_layer = 33 if "t33" in args.model else 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using model: {args.model}, repr_layer={repr_layer}, device={device}")

    fastas = collect_fastas(args.src_root)
    if not fastas:
        print("Not Found  FASTA。")
        sys.exit(1)

    entries_map: Dict[str, List[Tuple[str, str]]] = {}
    total_chains = 0
    for fa in fastas:
        entries = read_fasta(fa)
        entries_map[fa] = entries
        total_chains += len(entries)

    if total_chains == 0:
        sys.exit(0)

    from tqdm.auto import tqdm
    pbar = tqdm(total=total_chains, desc="Embedding chains", unit="chain")

    for fa in fastas:
        sys_name = system_name_from_path(fa)
        rel_dir  = os.path.relpath(os.path.dirname(fa), args.src_root)  # e.g. "1_BRD7_VHL"
        out_dir  = os.path.join(args.out, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        entries = entries_map[fa]
        if not entries:
            tqdm.write(f"[WARN] Blank FASTA,skip {fa}")
            continue

        for header, seq in entries:
            cid = parse_chain_id(header)
            safe_cid = cid if cid else "UNK"
            out_pt = os.path.join(out_dir, f"{sys_name}_chain_{safe_cid}__{args.model}.pt")

            if args.skip_existing and os.path.exists(out_pt):
                tqdm.write(f"[SKIP] Exist:{out_pt}")
                pbar.update(1)
                continue

            try:
                per_res, mean = esm_embed_sequence(seq, model, alphabet, device, repr_layer)

                payload: Dict[str, object] = {
                    "system": sys_name,
                    "chain_id": cid,
                    "seq": seq,
                    "model": args.model,
                    "repr_layer": repr_layer,
                    "mean": torch.from_numpy(mean),  # [d]
                }
                if not args.mean_only:
                    payload["per_residue"] = torch.from_numpy(per_res)  # [L,d]

                torch.save(payload, out_pt, _use_new_zipfile_serialization=True)
                tqdm.write(f"[OK] {sys_name} chain {cid}: {out_pt}  (L={len(seq)})")
            except Exception as e:
                tqdm.write(f"[ERROR] {sys_name} chain {cid}: {e}")
            pbar.set_postfix_str(f"{sys_name}|{cid}")
            pbar.update(1)

    pbar.close()
    print("[DONE] All Done")

if __name__ == "__main__":
    main()
