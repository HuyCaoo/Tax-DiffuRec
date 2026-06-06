"""
taxonomy_loss.py  (v3 — redesign hoàn toàn)
============================================
Các thay đổi căn bản so với v2:
  1. Thay BCE bằng Focal Loss để xử lý class imbalance multi-label
  2. Hard Negative Mining thay vì random negative cho Triplet Loss
  3. Thêm Contrastive Regularization: kéo rep_diffu gần embedding
     của ground-truth item hơn trong không gian L2
  4. Cat_Acc dùng per-sample top-1 category thay vì Hamming
     để phản ánh thực tế hơn

Architecture:
  L_taxonomy = beta * L_focal_cat + (1-beta) * L_hard_triplet
             + gamma * L_contrastive_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from collections import defaultdict


class FocalBCELoss(nn.Module):
    """
    Focal Loss cho multi-label classification.
    Giảm weight của easy negatives (label=0, pred~0),
    tập trung vào hard positives.
    gamma=2 là giá trị chuẩn từ paper gốc.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = 5.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight  # up-weight positive labels

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction='none'
        )
        # Focal weight: giảm loss của easy samples
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class TaxonomyLoss(nn.Module):
    """
    Taxonomy Loss v3:
      L_taxonomy = beta * L_focal_cat
                 + (1-beta) * L_hard_triplet
                 + gamma_reg * L_contrastive_reg

    Parameters
    ----------
    item_num      : int
    category_map  : dict — item_id -> int hoặc list[int]
    hidden_size   : int
    alpha         : float — trọng số tổng thể
    beta          : float — tỉ lệ focal_cat vs hard_triplet
    margin        : float — margin triplet
    gamma_reg     : float — trọng số contrastive regularization
    loss_scale    : float — scale để đưa về magnitude của loss_item
    focal_gamma   : float — focal loss gamma (mặc định 2.0)
    hard_k        : int   — số negative candidates để chọn hard negative
    device        : str
    multi_label   : bool
    """

    def __init__(
        self,
        item_num: int,
        category_map: Optional[Dict],
        hidden_size: int,
        alpha: float = 0.1,
        beta: float = 0.6,
        margin: float = 0.8,
        gamma_reg: float = 0.1,
        loss_scale: float = 10.0,
        focal_gamma: float = 2.0,
        hard_k: int = 10,
        device: str = 'cuda',
        multi_label: bool = False,
    ):
        super(TaxonomyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.gamma_reg = gamma_reg
        self.loss_scale = loss_scale
        self.hard_k = hard_k
        self.device = device
        self.multi_label = multi_label
        self.enabled = (category_map is not None)

        if not self.enabled:
            print("[TaxonomyLoss v3] category_map=None → Disabled.")
            return

        # ── 1. item→primary_category tensor ───────────────────────────────
        all_cats = set()
        for v in category_map.values():
            if isinstance(v, list):
                all_cats.update(v)
            else:
                all_cats.add(v)
        self.num_categories = max(all_cats) + 1

        item2cat = torch.zeros(item_num, dtype=torch.long)
        for item_id, cat in category_map.items():
            if item_id < item_num:
                primary = cat[0] if isinstance(cat, list) else cat
                item2cat[item_id] = int(primary)
        self.register_buffer('item2cat', item2cat)

        # ── 2. Multi-label mask ────────────────────────────────────────────
        ml_mask = torch.zeros(item_num, self.num_categories)
        for item_id, cat in category_map.items():
            if item_id < item_num:
                cats = cat if isinstance(cat, list) else [cat]
                for c in cats:
                    ml_mask[item_id, int(c)] = 1.0
        self.register_buffer('item2cat_multilabel', ml_mask)

        # ── 3. cat2items lookup ────────────────────────────────────────────
        cat2items_raw = defaultdict(list)
        for item_id in range(item_num):
            c = item2cat[item_id].item()
            cat2items_raw[c].append(item_id)
        self.cat2items = {
            c: torch.tensor(ids, dtype=torch.long)
            for c, ids in cat2items_raw.items()
        }
        self.all_item_ids = torch.arange(1, item_num, dtype=torch.long)

        # ── 4. Projection head: H → num_categories ────────────────────────
        self.category_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, self.num_categories)
        )

        # ── 5. Loss functions ──────────────────────────────────────────────
        self.focal_loss = FocalBCELoss(gamma=focal_gamma, pos_weight=5.0)
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')

        print(
            f"[TaxonomyLoss] ✅ v3 | num_cat={self.num_categories} | "
            f"alpha={alpha} | beta={beta} | margin={margin} | "
            f"gamma_reg={gamma_reg} | hard_k={hard_k} | "
            f"focal_gamma={focal_gamma} | loss_scale={loss_scale}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _get_cat_labels(self, item_labels):
        return self.item2cat[item_labels.squeeze(-1).long()]

    # ──────────────────────────────────────────────────────────────────────────
    # L1: Focal Category Loss (xử lý class imbalance)
    # ──────────────────────────────────────────────────────────────────────────
    def _loss_focal_cat(self, rep_diffu, item_labels):
        logits = self.category_proj(rep_diffu)           # (B, C)
        ids    = item_labels.squeeze(-1).long()
        target = self.item2cat_multilabel[ids]           # (B, C)
        return self.focal_loss(logits, target)

    # ──────────────────────────────────────────────────────────────────────────
    # L2: Hard Negative Triplet Loss
    # ──────────────────────────────────────────────────────────────────────────
    def _loss_hard_triplet(self, rep_diffu, item_embeddings, item_labels):
        """
        Hard negative: thay vì random, chọn negative có cosine similarity
        cao nhất với anchor trong số hard_k candidates ngẫu nhiên.
        """
        cat_labels = self._get_cat_labels(item_labels)
        pos_emb    = item_embeddings(item_labels.squeeze(-1))  # (B, H)

        neg_indices = self._sample_hard_negatives(
            rep_diffu.detach(), cat_labels, item_embeddings
        )
        neg_emb = item_embeddings(neg_indices)

        return self.triplet_loss(rep_diffu, pos_emb, neg_emb)

    def _sample_hard_negatives(self, anchor, cat_labels, item_embeddings):
        """
        Với mỗi sample, lấy hard_k candidates từ category khác,
        chọn candidate có distance nhỏ nhất với anchor (hardest negative).
        """
        device = cat_labels.device
        B = cat_labels.shape[0]
        neg_ids = torch.zeros(B, dtype=torch.long, device=device)
        anchor_norm = F.normalize(anchor, dim=-1)  # (B, H)

        unique_cats = cat_labels.unique().tolist()
        for cat_i in unique_cats:
            mask = (cat_labels == cat_i)
            n    = mask.sum().item()
            if n == 0:
                continue

            # Gộp items từ các category khác
            neg_pool = []
            for c, items in self.cat2items.items():
                if c != int(cat_i):
                    neg_pool.append(items)
            if not neg_pool:
                rand_idx = torch.randint(0, len(self.all_item_ids), (n,))
                neg_ids[mask] = self.all_item_ids[rand_idx]
                continue

            neg_pool_tensor = torch.cat(neg_pool).to(device)

            # Sample hard_k candidates ngẫu nhiên
            k = min(self.hard_k, len(neg_pool_tensor))
            rand_idx = torch.randint(0, len(neg_pool_tensor), (n * k,), device=device)
            cand_ids = neg_pool_tensor[rand_idx].reshape(n, k)  # (n, k)

            # Lấy embedding của candidates
            cand_ids_flat = cand_ids.reshape(-1).to(device)
            cand_emb = item_embeddings(cand_ids_flat)           # (n*k, H)
            cand_emb = F.normalize(cand_emb, dim=-1).reshape(n, k, -1)

            # Cosine similarity với anchor → chọn hardest (similarity cao nhất)
            anchor_i = anchor_norm[mask].unsqueeze(2)           # (n, H, 1)
            sim = torch.bmm(cand_emb, anchor_i).squeeze(-1)     # (n, k)
            hardest_idx = sim.argmax(dim=-1)                    # (n,)

            selected = cand_ids[
                torch.arange(n, device=device),
                hardest_idx.to(device)
            ]
            neg_ids[mask] = selected

        return neg_ids.to(device)

    # ──────────────────────────────────────────────────────────────────────────
    # L3: Contrastive Regularization
    # Kéo rep_diffu gần embedding của ground-truth item
    # ──────────────────────────────────────────────────────────────────────────
    def _loss_contrastive_reg(self, rep_diffu, item_embeddings, item_labels):
        """
        Cosine similarity loss: maximize sim(rep_diffu, item_emb_gt)
        """
        gt_emb = item_embeddings(item_labels.squeeze(-1))  # (B, H)
        rep_norm = F.normalize(rep_diffu, dim=-1)
        gt_norm  = F.normalize(gt_emb, dim=-1)
        cos_sim  = (rep_norm * gt_norm).sum(dim=-1)        # (B,)
        # loss = 1 - mean(cos_sim) → minimize khi sim cao
        return (1.0 - cos_sim).mean()

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, rep_diffu, item_labels, item_embeddings):
        if not self.enabled:
            return torch.tensor(0.0, device=rep_diffu.device, requires_grad=False)

        loss_cat  = self._loss_focal_cat(rep_diffu, item_labels)
        loss_tri  = self._loss_hard_triplet(rep_diffu, item_embeddings, item_labels)
        loss_reg  = self._loss_contrastive_reg(rep_diffu, item_embeddings, item_labels)

        loss_taxonomy = (
            self.beta * loss_cat
            + (1.0 - self.beta) * loss_tri
            + self.gamma_reg * loss_reg
        )

        return self.alpha * self.loss_scale * loss_taxonomy

    # ──────────────────────────────────────────────────────────────────────────
    # Category accuracy: top-1 predicted category vs primary category
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def category_accuracy(self, rep_diffu, item_labels) -> float:
        if not self.enabled:
            return 0.0
        logits   = self.category_proj(rep_diffu)
        pred_cat = logits.argmax(dim=-1)           # top-1 predicted category
        gt_cat   = self._get_cat_labels(item_labels)
        return (pred_cat == gt_cat).float().mean().item()
