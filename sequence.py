#!/usr/bin/env python3
import os, sys, glob, argparse
from typing import Dict, List, Tuple
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1



def one_letter(resname: str) -> str:
    rn = resname.strip().upper()
    return seq1(rn, undef_code="X")

def chain_sequence(chain) -> str:
    seq = []
    for res in chain.get_residues():
        if not is_aa(res, standard=False):
            continue
        seq.append(one_letter(res.get_resname()))
    return "".join(seq)

def extract_from_pdb(pdb_path: str) -> Dict[str, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    seqs = {}
    for model in structure:
        for chain in model:
            s = chain_sequence(chain)
            if len(s) >= 20:  
                seqs[str(chain.id)] = s
    return seqs

def write_fasta(out_path: str, entries: List[Tuple[str, str]], title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for cid, seq in entries:
            f.write(f">{title}|chain_{cid}|len={len(seq)}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

def find_first_merge(system_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(system_dir, "**", "merge.pdb"), recursive=True))
    return files[0] if files else ""

def main(src_root: str, out_root: str):
    if not os.path.isdir(src_root):
        print(f"Source directory not exist{src_root}")
        sys.exit(1)

    systems = [os.path.join(src_root, d) for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    systems.sort()

    if not systems:
        print("Not found")
        sys.exit(1)

    for sys_dir in systems:
        sys_name = os.path.basename(sys_dir)
        merge_pdb = find_first_merge(sys_dir)
        if not merge_pdb:
            print(f"[WARN] {sys_name}: not found merge_*.pdb, skip")
            continue

        seqs = extract_from_pdb(merge_pdb)
        if not seqs:
            print(f"[WARN] {sys_name}: {os.path.relpath(merge_pdb, sys_dir)} not found protein seq")
            continue

        sorted_chains = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)
        top2 = sorted_chains[:2]
        if len(top2) < 2:
            print(f"[WARN] {sys_name}:only {len(top2)} chain")

        rel_path = os.path.relpath(sys_dir, src_root)  # e.g., "1_BRD7_VHL"
        dest_dir = os.path.join(out_root, rel_path)
        out_fasta = os.path.join(dest_dir, f"{sys_name}.fasta")

        write_fasta(out_fasta, top2, title=sys_name)
        lens = ", ".join([f"{cid}:{len(seq)}aa" for cid, seq in top2])
        print(f"[OK] {sys_name}: WRITING {out_fasta}  ({lens}) from {os.path.relpath(merge_pdb, sys_dir)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract two protein-chain sequences from merge_*.pdb and mirror output structure.")
    ap.add_argument("src_root")
    ap.add_argument("--out", required=True, dest="out_root")
    args = ap.parse_args()
    main(args.src_root, args.out_root)
