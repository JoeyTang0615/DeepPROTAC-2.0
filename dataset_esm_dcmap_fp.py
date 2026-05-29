from typing import Any, Dict, List, Optional
import random
import torch
from torch.utils.data import Dataset

def _pad_1d_long(seqs: List[List[int]], pad_val: int = 0):
    if len(seqs) == 0:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.bool)
    max_len = max(len(s) for s in seqs)
    B = len(seqs)
    out = torch.full((B, max_len), pad_val, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = len(s)
        if L > 0:
            out[i, :L] = torch.tensor(s, dtype=torch.long)
            mask[i, :L] = True
    return out, mask

def _stack_bitvectors(maybe_vecs: List[Optional[torch.Tensor]], dtype: torch.dtype = torch.uint8):

    B = len(maybe_vecs)
    L = 0
    for v in maybe_vecs:
        if isinstance(v, torch.Tensor):
            L = int(v.numel())
            break
    if L == 0:
        return torch.empty(B, 0, dtype=dtype)

    out = torch.zeros((B, L), dtype=dtype)
    for i, v in enumerate(maybe_vecs):
        if isinstance(v, torch.Tensor):
            if v.numel() == L:
                out[i] = v.to(dtype)
            else:
                n = min(L, int(v.numel()))
                out[i, :n] = v.view(-1)[:n].to(dtype)
    return out

class PROTACBagDataset(Dataset):

    def __init__(
        self,
        packed: Dict[str, Any],
        filter_empty: bool = False,
        shuffle_within_name_each_epoch: bool = False,
        rng_seed: int = 1234,
        esm_fallback_dim: int = 1280,  
    ):
        self.shuffle_within = shuffle_within_name_each_epoch
        self.rng = random.Random(rng_seed)

        names   = list(map(str, packed["names"]))
        cg_all  = packed["contact_graphs"]
        pg_all  = packed["protac_graphs"]
        cm_all  = packed["cmap"]
        dmap_all= packed.get("dmap", [None for _ in names])          # 
        smis    = packed["protac_smi"]
        maccs   = packed.get("protac_maccs", [None for _ in names])  # 
        ecfp    = packed.get("protac_ecfp",  [None for _ in names])  # 
        esms    = packed.get("esm", [[] for _ in names])
        l1      = packed["label1"]
        l2      = packed["label2"]

        assert len(names)==len(cg_all)==len(pg_all)==len(cm_all)==len(smis)==len(esms)==len(l1)==len(l2), \
            "names/cg/pg/cm/smis/esm/label1/label2)"
        assert len(dmap_all)==len(names), "dmap length not match"
        assert len(maccs)==len(names) and len(ecfp)==len(names), "maccs/ecfp not match"

        esm_dim = None
        for es in esms:
            if isinstance(es, (list, tuple)) and len(es) > 0 and isinstance(es[0], torch.Tensor):
                esm_dim = int(es[0].numel())
                break
        if esm_dim is None:
            esm_dim = esm_fallback_dim

        self.names, self.cg, self.pg, self.cm = [], [], [], []
        self.smis, self.esm, self.l1, self.l2 = [], [], [], []
        self.dmap, self.maccs, self.ecfp      = [], [], []

        for i in range(len(names)):
            if filter_empty:
                is_all_empty = (len(cg_all[i]) == 0) and (len(pg_all[i]) == 0) and (len(cm_all[i]) == 0)
                if is_all_empty:
                    continue

            esm_list = list(esms[i])
            if len(esm_list) < 2:
                if len(esm_list) == 1 and isinstance(esm_list[0], torch.Tensor):
                    z = torch.zeros_like(esm_list[0])
                    esm_list = [esm_list[0], z]
                else:
                    z = torch.zeros(esm_dim, dtype=torch.float32)
                    esm_list = [z.clone(), z.clone()]
            elif len(esm_list) > 2:
                esm_list = esm_list[:2]

            self.names.append(names[i])
            self.cg.append(list(cg_all[i]))
            self.pg.append(list(pg_all[i]))
            self.smis.append(list(smis[i]))  
            self.esm.append([e.float().view(-1) for e in esm_list])
            self.l1.append(int(l1[i]))
            self.l2.append(int(l2[i]))
            self.dmap.append(dmap_all[i] if isinstance(dmap_all[i], torch.Tensor) else None)
            self.cm.append(cm_all[i] if isinstance(cm_all[i], torch.Tensor) else None)
            self.maccs.append(maccs[i] if isinstance(maccs[i], torch.Tensor) else None)
            self.ecfp.append(ecfp[i] if isinstance(ecfp[i], torch.Tensor) else None)

    def __len__(self):
        return len(self.names)

    def on_epoch_start(self):

        if not self.shuffle_within:
            return
        for i in range(len(self.names)):
            self.rng.shuffle(self.cg[i])
            self.rng.shuffle(self.pg[i])
            self.rng.shuffle(self.cm[i])

    def __getitem__(self, idx: int):
        return {
            "name": self.names[idx],
            "contact_graphs": self.cg[idx],
            "protac_graphs":  self.pg[idx],
            "cmap":   self.cm[idx],
            "dmap":           self.dmap[idx],      # Tensor or None
            "protac_smi":     self.smis[idx],
            "protac_maccs":   self.maccs[idx],     # Tensor(uint8) or None
            "protac_ecfp":    self.ecfp[idx],      # Tensor(uint8) or None
            "esm_means":      torch.stack(self.esm[idx], dim=0),  # [2, D]
            "label1":         self.l1[idx],
            "label2":         self.l2[idx],
        }

def protac_bag_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:

    names = [b["name"] for b in batch]
    l1 = torch.tensor([b["label1"] for b in batch], dtype=torch.long)
    l2 = torch.tensor([b["label2"] for b in batch], dtype=torch.long)

    # SMILES padding
    smi_list = [b["protac_smi"] for b in batch]         # List[List[int]]
    smi_ids, smi_mask = _pad_1d_long(smi_list, pad_val=0)

    # ESM mean 
    esm_means = torch.stack([b["esm_means"] for b in batch], dim=0)  # [B, 2, D]

    # MACCS / ECFP 
    maccs = _stack_bitvectors([b["protac_maccs"] for b in batch], dtype=torch.uint8)  # [B, Lm]
    ecfp  = _stack_bitvectors([b["protac_ecfp"]  for b in batch], dtype=torch.uint8)  # [B, Le]

    # DMAP
    dmaps = [b["dmap"] for b in batch]
    dmap_n = [int(x.shape[0]) if isinstance(x, torch.Tensor) else 0 for x in dmaps]

    return {
        "name": names,
        "label1": l1,
        "label2": l2,
        "smi_ids": smi_ids,         # [B, T]
        "smi_mask": smi_mask,       # [B, T]
        "esm_means": esm_means,     # [B, 2, D]
        "maccs": maccs,             # [B, Lm]  (Lm=167 )
        "ecfp": ecfp,               # [B, Le]  (Le=2048 )
        "dmap": dmaps,              # List[Tensor or None]
        "dmap_n": dmap_n,           # List[int]
        "contact_graphs": [b["contact_graphs"] for b in batch],
        "protac_graphs":  [b["protac_graphs"]  for b in batch],
        "cmap":           [b["cmap"]   for b in batch],
    }
