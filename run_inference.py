# -*- coding: utf-8 -*-

import os
import glob
import pickle
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
except ImportError:
    raise ImportError(" 'pip install rdkit'")

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from torch_geometric.data import Data

from dataset_esm_dcmap_fp import PROTACBagDataset, protac_bag_collate
from model_refine6 import TriComplexPredictor


def one_letter(resname: str) -> str:
    rn = resname.strip().upper()
    return seq1(rn, undef_code="X")

def extract_sequences_from_pdb(pdb_path: str) -> List[str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    seqs = {}
    for model in structure:
        for chain in model:
            seq_list = []
            for res in chain.get_residues():
                if not is_aa(res, standard=False):
                    continue
                seq_list.append(one_letter(res.get_resname()))
            s = "".join(seq_list)
            if len(s) >= 20:
                seqs[str(chain.id)] = s
                
    sorted_chains = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)
    top2_seqs = [seq for cid, seq in sorted_chains[:2]]
    return top2_seqs

@torch.no_grad()
def get_real_esm_embedding(seq: str, model, alphabet, device, repr_layer: int, max_tokens: int = 1022) -> torch.Tensor:
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    L = len(seq)
    
    def _embed_chunk(subseq: str):
        data = [("seq", subseq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]
        reps = reps[0, 1:1+len(subseq), :]
        return reps.detach().cpu()

    if L <= max_tokens:
        per_res = _embed_chunk(seq)
        return per_res.mean(dim=0)
        
    chunks = []
    for start in range(0, L, max_tokens):
        sub = seq[start: start + max_tokens]
        chunks.append(_embed_chunk(sub))
    per_res = torch.cat(chunks, dim=0)[:L]
    return per_res.mean(dim=0)


class EndToEndRealDataPipeline:
    def __init__(self, data_root_dir: str, esm_model_name: str = "esm2_t33_650M_UR50D"):
        self.data_root = data_root_dir
        self.pdb_parser = PDBParser(QUIET=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[ESM-Init] Load {esm_model_name} ")
        import esm
        load_fn = getattr(esm.pretrained, esm_model_name)
        self.esm_model, self.esm_alphabet = load_fn()
        self.esm_model = self.esm_model.to(self.device).eval()
        self.repr_layer = self.esm_model.num_layers
        
    def process_and_pack_all(self, output_pkl_path: str):

        system_dirs = sorted([
            os.path.join(self.data_root, d) 
            for d in os.listdir(self.data_root) 
            if os.path.isdir(os.path.join(self.data_root, d)) and not d.endswith('.ipynb_checkpoints')
        ])
        
        packed = {
            "names": [], "contact_graphs": [], "protac_graphs": [], "cmap": [], 
            "dmap": [], "protac_smi": [], "protac_maccs": [], "protac_ecfp": [], 
            "esm": [], "label1": [], "label2": [], "vocab_size": 128
        }
        

        for s_dir in system_dirs:
            sys_name = os.path.basename(s_dir.rstrip("/\\"))
            merge_pdb = os.path.join(s_dir, "merge.pdb")
            sdf_files = glob.glob(os.path.join(s_dir, "*_confs_*.sdf"))
            
            if not os.path.exists(merge_pdb) or not sdf_files:
                continue
                
            
            protein_seqs = extract_sequences_from_pdb(merge_pdb)
            if not protein_seqs: continue
                
            esm_vectors = []
            for seq in protein_seqs:
                v_mean = get_real_esm_embedding(seq, self.esm_model, self.esm_alphabet, self.device, self.repr_layer)
                esm_vectors.append(v_mean)

            mol = Chem.SDMolSupplier(sdf_files[0])[0]
            if mol is None: continue
            smi = Chem.MolToSmiles(mol)
            protac_smi_tokens = [ord(c) % 120 + 1 for c in smi]
            
            maccs_bits = [int(b) for b in MACCSkeys.GenMACCSKeys(mol).ToBitString()]
            ecfp_bits = [int(b) for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()]
            
            pg_x = torch.tensor([atom.GetAtomicNum() % 10 for atom in mol.GetAtoms()], dtype=torch.long)
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                b_type = int(bond.GetBondTypeAsDouble()) % 6
                edge_indices.extend([[u, v], [v, u]])
                edge_attrs.extend([b_type, b_type])
            pg_edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
            pg_edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
            pg_pos = torch.tensor([list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float32)
            protac_graph = Data(x=pg_x, edge_index=pg_edge_index, edge_attr=pg_edge_attr, pos=pg_pos)

            structure = self.pdb_parser.get_structure(sys_name, merge_pdb)
            ca_atoms = [res['CA'] for model in structure for chain in model for res in chain.get_residues() if 'CA' in res]
            coords = torch.tensor(np.array([atom.get_coord() for atom in ca_atoms]), dtype=torch.float32)
            dmap = torch.cdist(coords, coords)
            cmap = (dmap < 12.0).float()
            
            cg_x = torch.tensor([[ord(a.get_parent().get_resname()[0]) % 23, ord(a.get_parent().get_parent().id) % 7, i % 24] for i, a in enumerate(ca_atoms)], dtype=torch.long)
            cg_edges, cg_edge_attrs = [], []
            for u in range(len(ca_atoms)):
                for v in range(u + 1, len(ca_atoms)):
                    if dmap[u, v].item() < 8.0:
                        cg_edges.extend([[u, v], [v, u]]); cg_edge_attrs.extend([1, 1])
            cg_edge_index = torch.tensor(cg_edges, dtype=torch.long).t().contiguous() if cg_edges else torch.empty((2, 0), dtype=torch.long)
            cg_edge_attr = torch.tensor(cg_edge_attrs, dtype=torch.long)
            contact_graph = Data(x=cg_x, edge_index=cg_edge_index, edge_attr=cg_edge_attr, pos=coords)

            packed["names"].append(sys_name)
            packed["contact_graphs"].append([contact_graph])
            packed["protac_graphs"].append([protac_graph])
            packed["cmap"].append(cmap)
            packed["dmap"].append(dmap)
            packed["protac_smi"].append(protac_smi_tokens)
            packed["protac_maccs"].append(torch.tensor(maccs_bits, dtype=torch.uint8))
            packed["protac_ecfp"].append(torch.tensor(ecfp_bits, dtype=torch.uint8))
            packed["esm"].append(esm_vectors)
            packed["label1"].append(0)
            packed["label2"].append(0)
            
        with open(output_pkl_path, "wb") as f:
            pickle.dump(packed, f)
        print(f"\n Write to -> {output_pkl_path}\n")
        return output_pkl_path

    @torch.no_grad()
    def execute_inference(self, pkl_data_path: str, checkpoint_path: str):

        
        model = TriComplexPredictor(

            vocab_size=128,                     
            hidden_dim=64,                     
            num_layers_gnn=3,                   
            num_classes1=1,
            num_classes2=1,
            

            num_node_embeddings_protac=10,      
            num_edge_embeddings_protac=6,      
            protein_node_field_sizes=[23, 7, 24],
            num_edge_embeddings_protein=6,      
            gnn_heads=4,                        
            maccs_dim=167,                      
            ecfp_dim=2048,                      
            
            esm_dim=1280,                       
            esm_num_layers=5,                   
            esm_attn_heads=8,                  

            seq_dropout=0.0, 
            seq_tokendrop=0.0, 
            gnn_dropout=0.0, 
            gnn_dropedge=0.0, 
            cm_drop2d=0.0, 
            fuse_dropout=0.0,
            esm_dropout=0.0
        ).to(self.device)
        
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f" {checkpoint_path}")
        else:
            print(f"Not found ckp")
            
        model.eval()
        
        with open(pkl_data_path, "rb") as f:
            packed_data = pickle.load(f)
            
        dataset = PROTACBagDataset(packed_data, filter_empty=False, shuffle_within_name_each_epoch=False)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=protac_bag_collate)
        
        print("\n" + "="*40 + "\nResults:\n" + "="*40)
        for batch in loader:
            outputs = model(batch)
            
            probs1 = torch.sigmoid(outputs["logits1"]).cpu().flatten().tolist()
            probs2 = torch.sigmoid(outputs["logits2"]).cpu().flatten().tolist()
            
            for idx, name in enumerate(batch["name"]):
                print(f"System : {name}")
                print(f"  Label 1: {probs1[idx]:.4f} (: {' Active' if probs1[idx] >= 0.5 else ' Inactive'})")
                print("-" * 30)


if __name__ == "__main__":
    RAW_DATA_DIR = "./data" 
    PKL_OUTPUT_PATH = "./demo_real_esm_dataset.pkl"
    CHECKPOINT_PATH = "./ckp/checkpoint.pt"
    
    if os.path.exists(RAW_DATA_DIR):
        flow = EndToEndRealDataPipeline(data_root_dir=RAW_DATA_DIR, esm_model_name="esm2_t33_650M_UR50D")
        pkl_path = flow.process_and_pack_all(output_pkl_path=PKL_OUTPUT_PATH)
        flow.execute_inference(pkl_data_path=pkl_path, checkpoint_path=CHECKPOINT_PATH)
    else:
        print(f"Not found '{RAW_DATA_DIR}' ")