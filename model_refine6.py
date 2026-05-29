# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, PointNetConv, global_mean_pool


# ---------------------- Small Helpers ----------------------
def _warn(msg: str):
    try:
        import warnings
        warnings.warn(msg)
    except Exception:
        pass


class MaybeEmbedding(nn.Module):
    """If input is long/int -> Embedding; if float tensor -> Linear projection to H."""
    def __init__(self, in_dim_or_num_embeddings: Optional[int], hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_embeddings = in_dim_or_num_embeddings
        if in_dim_or_num_embeddings is None:
            self.is_discrete = False  # expect float features
            self.proj = None  # determined lazily if needed
        else:
            self.is_discrete = True
            self.emb = nn.Embedding(in_dim_or_num_embeddings, hidden_dim)

    def _encode_float_vector(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        in_f = x.size(-1)
        if not hasattr(self, 'proj') or self.proj is None or getattr(self, '_proj_in_f', None) != in_f:
            self.proj = nn.Linear(in_f, self.hidden_dim)
            self._proj_in_f = in_f
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            return self.emb(x)
        else:
            return self._encode_float_vector(x)


class TransformerSeqEncoder(nn.Module):
    """
    Token + positional embedding -> TransformerEncoder -> masked mean pooling.
    Added regularization: embedding dropout and token_drop (randomly drop valid tokens at training).
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.2,
        token_drop: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.token_drop = float(token_drop)

    def forward(self, smi_ids: torch.Tensor, smi_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # smi_ids: [B, T], smi_mask: [B, T] [True for valid]
        B, T = smi_ids.shape
        pos = torch.arange(T, device=smi_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(smi_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        if smi_mask is None:
            valid = torch.ones_like(smi_ids, dtype=torch.bool)
        else:
            valid = smi_mask

        # TokenDrop: randomly drop some valid tokens during training
        if self.training and self.token_drop > 0.0:
            drop = (torch.rand_like(valid.float()) < self.token_drop) & valid
            keep = valid & (~drop)
            # avoid empty sequence
            row_keep = keep.any(dim=1, keepdim=True)
            keep = torch.where(row_keep, keep, valid)
            x = x * keep.unsqueeze(-1)
            valid = keep

        z = self.encoder(x, src_key_padding_mask=~valid)  # [B, T, H]

        valid_f = valid.float()
        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (z * valid_f.unsqueeze(-1)).sum(dim=1) / denom
        return self.norm(pooled)  # [B, H]


# ---------------------- Graph Encoder (TransformerConv + PointNetConv) ----------------------
class TransPointBlock(nn.Module):
    """
    一个 block = TransformerConv (+ edge_attr)  +  PointNetConv(pos 上的 PointNet 卷积)
    如果图里没有 pos,就自动退化成只用 TransformerConv。
    """
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: Optional[int] = None,
        use_pointnet: bool = True,
    ):
        super().__init__()
        self.use_pointnet = use_pointnet
        self.trans = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,        # keep dim = hidden_dim
            dropout=dropout,
            edge_dim=edge_dim,
        )
        if use_pointnet:
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.point = PointNetConv(local_nn=local_nn, global_nn=None)
        else:
            self.point = None

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is not None:
            x_t = self.trans(x, edge_index, edge_attr=edge_attr)
        else:
            x_t = self.trans(x, edge_index)

        if self.point is not None and pos is not None:
            x_p = self.point(x, pos, edge_index)
            h = x_t + x_p
        else:
            h = x_t

        h = self.norm(h)
        h = self.dropout(h)
        return h


class GraphEncoder(nn.Module):
    """
    Stacks multiple TransPoint blocks; supports:
      - node: long indices -> Embedding; float vectors -> Linear
      - edge: long indices -> Embedding; float vectors -> Linear
    Adds per-layer stochastic DropEdge during training.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_node_embeddings: Optional[int] = None,
        num_edge_embeddings: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.3,
        dropedge_p: float = 0.1,
        trans_heads: int = 4,
        use_pointnet: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropedge_p = float(dropedge_p)
        self.node_enc = MaybeEmbedding(num_node_embeddings, hidden_dim)
        self.edge_enc = MaybeEmbedding(num_edge_embeddings, hidden_dim)

        self.blocks = nn.ModuleList([
            TransPointBlock(
                hidden_dim,
                heads=trans_heads,
                dropout=dropout,
                edge_dim=hidden_dim if num_edge_embeddings is not None else None,
                use_pointnet=use_pointnet,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def _maybe_encode_node(self, x: torch.Tensor) -> torch.Tensor:
        # If x is long indices or float vectors
        if x.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
            return self.node_enc(x)
        else:
            # float
            return self.node_enc._encode_float_vector(x)

    def _maybe_encode_edge(self, e: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if e is None:
            return None
        if e.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
            return self.edge_enc(e)
        else:
            return self.edge_enc._encode_float_vector(e)

    def _safe_forward_single_graph(self, g: Data) -> Optional[torch.Tensor]:
        try:
            x = g.x
            eidx = g.edge_index
            eattr = getattr(g, "edge_attr", None)
            pos = getattr(g, "pos", None)

            x = self._maybe_encode_node(x)
            eattr = self._maybe_encode_edge(eattr)
            if pos is not None:
                pos = pos.to(x.device).float()

            # init node hidden
            h = x
            # per-layer conv + DropEdge (training only)
            for blk in self.blocks:
                ei = eidx
                ea = eattr
                if self.training and self.dropedge_p > 0.0 and ei is not None and ei.numel() > 0:
                    num_edges = ei.size(1)
                    keep_mask = torch.rand(num_edges, device=ei.device) >= self.dropedge_p
                    if keep_mask.sum() == 0:
                        keep_mask[torch.randint(0, num_edges, (1,), device=ei.device)] = True
                    ei = ei[:, keep_mask]
                    if ea is not None:
                        if ea.dim() == 1:
                            ea = ea[keep_mask]
                        else:
                            ea = ea[keep_mask, :]
                h = blk(h, ei, pos, ea)

            # simple global mean pool (graph-level vector)
            if hasattr(g, "batch") and g.batch is not None:
                pooled = global_mean_pool(h, g.batch)
            else:
                pooled = h.mean(dim=0, keepdim=True)
            v = self.norm(pooled.squeeze(0))
            return v
        except Exception as ex:
            _warn(f"skip one graph due to: {ex}")
            return None

    def forward(self, graphs: List[Data]) -> torch.Tensor:
        vecs = []
        for g in graphs:
            v = self._safe_forward_single_graph(g)
            if v is not None and torch.isfinite(v).all():
                vecs.append(v)

        if len(vecs) == 0:
            device = next(self.parameters()).device
            return torch.zeros(self.hidden_dim, device=device)

        Z = torch.stack(vecs, dim=0)  # [K_valid, H]
        return Z.mean(dim=0)


# ProteinGraphEncoder: reuse GraphEncoder but replace node encoder with multi-field capable encoder
class ProteinNodeEncoder(nn.Module):
    """Support node x as multi-field long indices [N,C] or float vectors [N,F]."""
    def __init__(self, hidden_dim: int, field_sizes: Optional[List[int]] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.field_sizes = field_sizes  # e.g., [E, B, R, ...]; if None -> lazy
        self.embs: nn.ModuleList = nn.ModuleList()
        self.linear: Optional[nn.Linear] = None

        if field_sizes is not None:
            for n in field_sizes:
                self.embs.append(nn.Embedding(n, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
            # long indices: shape [N, C]
            if x.dim() == 1:
                x = x.unsqueeze(1)
            N, C = x.shape
            # lazy-create embs if not provided
            if len(self.embs) == 0:
                # assume each column has max index + 1 vocab size
                sizes = []
                for c in range(C):
                    vmax = int(x[:, c].max().item())
                    sizes.append(max(1, vmax + 1))
                for n in sizes:
                    self.embs.append(nn.Embedding(n, self.hidden_dim))
            parts = []
            for c in range(x.size(1)):
                parts.append(self.embs[c](x[:, c]))
            h = sum(parts) / float(len(parts))
            return h
        else:
            # float features [N, F]
            if self.linear is None or self.linear.in_features != x.size(-1):
                self.linear = nn.Linear(x.size(-1), self.hidden_dim)
            return self.linear(x)


class ProteinGraphEncoder(GraphEncoder):
    """
    Protein graph encoder：node 用 ProteinNodeEncoder，conv 用 TransformerConv + PointNetConv。
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        node_field_sizes: Optional[List[int]] = None,
        num_edge_embeddings: Optional[int] = 6,
        num_layers: int = 3,
        dropout: float = 0.2,
        dropedge_p: float = 0.1,
        trans_heads: int = 4,
        use_pointnet: bool = True,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_node_embeddings=None,               # replaced below
            num_edge_embeddings=num_edge_embeddings,
            num_layers=num_layers,
            dropout=dropout,
            dropedge_p=dropedge_p,
            trans_heads=trans_heads,
            use_pointnet=use_pointnet,
        )

        self.node_enc = ProteinNodeEncoder(hidden_dim, field_sizes=node_field_sizes)


# ---------------------- Contact-Map Encoder ----------------------
class ContactMapEncoder(nn.Module):
    """
    Simple CNN stack for contact maps. Added spatial dropout (Dropout2d) for regularization.
    """
    def __init__(self, hidden_dim: int = 256, drop2d: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.drop2d = nn.Dropout2d(drop2d)
        self.proj = nn.Linear(128, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _forward_single_safe(self, cm: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if cm.dim() == 2:
                cm = cm.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            elif cm.dim() == 3:
                cm = cm.unsqueeze(0)               # [1,1,H,W] or [1,C,H,W]; expect single-channel
            if cm.size(1) != 1:
                cm = cm.mean(dim=1, keepdim=True)

            cm = cm.float()
            x = F.relu(self.conv1(cm)); x = self.drop2d(x); x = F.avg_pool2d(x, kernel_size=2)
            x = F.relu(self.conv2(x)); x = self.drop2d(x); x = F.avg_pool2d(x, kernel_size=2)
            x = F.relu(self.conv3(x)); x = self.drop2d(x)
            x = x.mean(dim=[2, 3])       # [B=1, 128]
            h = self.proj(x)             # [1, H]
            v = self.norm(h.squeeze(0))  # [H]
            return v
        except Exception:
            return None

    def forward(self, cms: List[torch.Tensor]) -> torch.Tensor:
        vecs = []
        for cm in cms:
            v = self._forward_single_safe(cm)
            if v is not None and torch.isfinite(v).all():
                vecs.append(v)

        if len(vecs) == 0:
            device = next(self.parameters()).device
            return torch.zeros(self.hidden_dim, device=device)

        Z = torch.stack(vecs, dim=0)  # [K_valid, H]
        return Z.mean(dim=0)


class ESMEncoder(nn.Module):
    """
    深度 ESM mean 特征编码模块
    输入: esm_means [B, 2, D]
    输出: esm_out [B, hidden_dim]
    """
    def __init__(
        self,
        esm_dim: int = 1280,
        hidden_dim: int = 512,
        num_layers: int = 3,
        attn_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        mlp_layers = []
        in_dim = esm_dim
        for _ in range(num_layers):
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, esm_means: torch.Tensor) -> torch.Tensor:
        """
        esm_means: [B, 2, D]
        """
        B, N, D = esm_means.shape
        assert N == 2, "Each sample needs two ESM mean encoding"

        x = self.mlp(esm_means)  # [B, 2, hidden_dim]

        attn_out, _ = self.cross_attn(x, x, x)  # [B, 2, hidden_dim]

        x = self.norm(x + attn_out)

        fused = x.mean(dim=1)  # [B, hidden_dim]

        out = self.proj_out(fused)  # [B, hidden_dim]
        return out


class FingerprintEncoder(nn.Module):
    """
    Encode PROTAC fingerprints (MACCS + ECFP) into a single [B, H] vector.
    Both inputs are optional; if one of them is None, we only use the other.
    """
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        maccs_dim: Optional[int] = None,
        ecfp_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.maccs_mlp: Optional[nn.Sequential] = None
        self.ecfp_mlp: Optional[nn.Sequential] = None
        self._maccs_in: Optional[int] = None
        self._ecfp_in: Optional[int] = None

        if maccs_dim is not None:
            self.maccs_mlp = self._build_mlp(maccs_dim)
            self._maccs_in = maccs_dim
        if ecfp_dim is not None:
            self.ecfp_mlp = self._build_mlp(ecfp_dim)
            self._ecfp_in = ecfp_dim

    def _build_mlp(self, in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        maccs: Optional[torch.Tensor],
        ecfp: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert (maccs is not None) or (ecfp is not None), "At least one fingerprint should be provided"

        device = maccs.device if maccs is not None else ecfp.device
        h_list = []

        if maccs is not None:
            maccs = maccs.float()
            if self.maccs_mlp is None or self._maccs_in != maccs.size(-1):
                self.maccs_mlp = self._build_mlp(maccs.size(-1)).to(device)
                self._maccs_in = maccs.size(-1)
            h_list.append(self.maccs_mlp(maccs))

        if ecfp is not None:
            ecfp = ecfp.float()
            if self.ecfp_mlp is None or self._ecfp_in != ecfp.size(-1):
                self.ecfp_mlp = self._build_mlp(ecfp.size(-1)).to(device)
                self._ecfp_in = ecfp.size(-1)
            h_list.append(self.ecfp_mlp(ecfp))

        # average the two if both exist
        if len(h_list) == 1:
            h = h_list[0]
        else:
            h = torch.stack(h_list, dim=0).mean(dim=0)  # [2,B,H] -> [B,H]
        return h


class TriComplexPredictor(nn.Module):
    """
    Sequence + (protein graph, PROTAC graph) + contact-map + ESM + PROTAC FP encoders -> fusion -> heads.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers_gnn: int = 3,
        num_classes1: int = 1,
        num_classes2: int = 1,
        num_node_embeddings_protac: Optional[int] = 32,
        num_edge_embeddings_protac: Optional[int] = 8,
        protein_node_field_sizes: Optional[List[int]] = None,  # e.g., [E, B, R, ...]
        num_edge_embeddings_protein: Optional[int] = 6,        # bond type classes (including unknown=0)
        seq_dropout: float = 0.1,
        seq_tokendrop: float = 0.0,
        gnn_dropout: float = 0.1,
        gnn_dropedge: float = 0.10,
        cm_drop2d: float = 0.10,
        fuse_dropout: float = 0.10,
        esm_dim: int = 1280,
        esm_num_layers: int = 3,
        esm_attn_heads: int = 8,
        esm_dropout: float = 0.2,
        gnn_heads: int = 4,
        maccs_dim: Optional[int] = None,
        ecfp_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.seq_enc = TransformerSeqEncoder(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            dropout=seq_dropout, token_drop=seq_tokendrop
        )

        # protein graph: TransformerConv + PointNetConv
        self.protein_enc = ProteinGraphEncoder(
            hidden_dim=hidden_dim,
            node_field_sizes=protein_node_field_sizes,
            num_edge_embeddings=num_edge_embeddings_protein,
            num_layers=num_layers_gnn,
            dropout=gnn_dropout,
            dropedge_p=gnn_dropedge,
            trans_heads=gnn_heads,
            use_pointnet=True,
        )

        # PROTAC graph: TransformerConv + PointNetConv
        self.protac_enc = GraphEncoder(
            hidden_dim=hidden_dim,
            num_node_embeddings=num_node_embeddings_protac,
            num_edge_embeddings=num_edge_embeddings_protac,
            num_layers=num_layers_gnn,
            dropout=gnn_dropout,
            dropedge_p=gnn_dropedge,
            trans_heads=gnn_heads,
            use_pointnet=True,
        )

        # contact map (use cmap from batch)
        self.cm_enc = ContactMapEncoder(hidden_dim=hidden_dim, drop2d=cm_drop2d)

        # ESM mean encoder 
        self.esm_enc = ESMEncoder(
            esm_dim=esm_dim,
            hidden_dim=hidden_dim,
            num_layers=esm_num_layers,
            attn_heads=esm_attn_heads,
            dropout=esm_dropout,
        )

        # PROTAC fingerprint encoder (MACCS + ECFP)
        self.fp_enc = FingerprintEncoder(
            hidden_dim=hidden_dim,
            dropout=fuse_dropout,
            maccs_dim=maccs_dim,
            ecfp_dim=ecfp_dim,
        )

        #  seq / protein / protac / cm / esm / fp => 6 * H
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(fuse_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head1 = nn.Linear(hidden_dim, num_classes1)
        self.head2 = nn.Linear(hidden_dim, num_classes2)

    # ---- internal 1-sample encode (called in forward for each sample) ----
    def _encode_one_sample(
        self,
        smi_ids_i: torch.Tensor,
        smi_mask_i: Optional[torch.Tensor],
        prot_graphs: List[Data],
        protac_graphs: List[Data],
        cms: List[torch.Tensor],
        esm_means_i: Optional[torch.Tensor],   # [2, D] or None
        maccs_i: Optional[torch.Tensor],
        ecfp_i: Optional[torch.Tensor],
    ) -> torch.Tensor:

        # 1) sequence
        h_seq = self.seq_enc(
            smi_ids_i.unsqueeze(0),
            smi_mask_i.unsqueeze(0) if smi_mask_i is not None else None
        )[0]  # [H]

        # 2) protein / protac graphs
        h_protein = self.protein_enc(prot_graphs)  # [H]
        h_protac  = self.protac_enc(protac_graphs) # [H]

        # 3) contact map
        h_cm = self.cm_enc(cms)                    # [H]

        # 4) ESM mean features 
        if esm_means_i is not None:
            if esm_means_i.dim() == 2:  # [2, D]
                h_esm = self.esm_enc(esm_means_i.unsqueeze(0))[0]  # [H]
            else:
                # 容错：若已是 [1,2,D]
                h_esm = self.esm_enc(esm_means_i)[0]
        else:
            device = next(self.parameters()).device
            h_esm = torch.zeros(self.hidden_dim, device=device)

        # 5) PROTAC fingerprints
        if maccs_i is not None or ecfp_i is not None:
            # add batch dimension to match FingerprintEncoder input expectations
            maccs_b = maccs_i.unsqueeze(0) if maccs_i is not None else None
            ecfp_b = ecfp_i.unsqueeze(0) if ecfp_i is not None else None
            h_fp = self.fp_enc(maccs_b, ecfp_b)[0]  # [H]
        else:
            device = next(self.parameters()).device
            h_fp = torch.zeros(self.hidden_dim, device=device)

        return torch.cat([h_seq, h_protein, h_protac, h_cm, h_esm, h_fp], dim=-1)  # [6H]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        smi_ids: torch.Tensor = batch["smi_ids"].to(device)              # [B, T]
        smi_mask: Optional[torch.Tensor] = batch.get("smi_mask", None)
        if smi_mask is not None:
            smi_mask = smi_mask.to(device)

        contact_graphs: List[List[Data]] = batch["contact_graphs"]       # protein graphs
        protac_graphs: List[List[Data]] = batch["protac_graphs"]
        cmaps: List[List[torch.Tensor]] = batch["cmap"]
        esm_means: Optional[torch.Tensor] = batch.get("esm_means", None)  # [B, 2, D]
        if esm_means is not None:
            esm_means = esm_means.to(device)

        maccs: Optional[torch.Tensor] = batch.get("maccs", None)         # [B, Lm]
        ecfp: Optional[torch.Tensor] = batch.get("ecfp", None)           # [B, Le]
        if maccs is not None:
            maccs = maccs.to(device)
        if ecfp is not None:
            ecfp = ecfp.to(device)

        B = smi_ids.size(0)
        fused_list = []
        for i in range(B):
            prot_graphs_i = []
            for g in contact_graphs[i]:
                g = g.clone()
                g.x = g.x.to(device)
                g.edge_index = g.edge_index.to(device)
                if getattr(g, "edge_attr", None) is not None:
                    g.edge_attr = g.edge_attr.to(device)
                if getattr(g, "pos", None) is not None:
                    g.pos = g.pos.to(device)
                prot_graphs_i.append(g)

            protac_graphs_i = []
            for g in protac_graphs[i]:
                g = g.clone()
                g.x = g.x.to(device)
                g.edge_index = g.edge_index.to(device)
                if getattr(g, "edge_attr", None) is not None:
                    g.edge_attr = g.edge_attr.to(device)
                if getattr(g, "pos", None) is not None:
                    g.pos = g.pos.to(device)
                protac_graphs_i.append(g)

            cms_i = [cm.to(device) for cm in cmaps[i]]

            esm_means_i = esm_means[i] if esm_means is not None else None  # [2, D] or None
            maccs_i = maccs[i] if maccs is not None else None
            ecfp_i = ecfp[i] if ecfp is not None else None

            fused = self._encode_one_sample(
                smi_ids[i],
                smi_mask[i] if smi_mask is not None else None,
                prot_graphs_i,
                protac_graphs_i,
                cms_i,
                esm_means_i,
                maccs_i,
                ecfp_i,
            )  # [6H]
            fused_list.append(fused)

        Fused = torch.stack(fused_list, dim=0)  # [B, 6H]
        H = self.fuse(Fused)                    # [B, H]
        logits1 = self.head1(H)                 # [B, C1]
        logits2 = self.head2(H)                 # [B, C2]
        return {"logits1": logits1, "logits2": logits2, "feat": H}
