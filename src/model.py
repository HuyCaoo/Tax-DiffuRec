"""
model.py  — DiffuRec + Category Embedding + Taxonomy Loss (Combined)
====================================================================
Kết hợp đồng thời:
  1. Category Embedding : e_i = e_item + alpha * e_cat  (input enrichment)
  2. Taxonomy Loss      : L_total = L_CE + L_taxonomy   (training signal)

Luồng:
  sequence → item_emb + cat_emb → LayerNorm → Transformer
           → Forward Diffusion → Reverse Denoising → rep_diffu
           → L_CE + L_taxonomy (Focal + Hard Triplet)
"""

import torch.nn as nn
import torch
from diffurec import DiffuRec
import torch.nn.functional as F
import numpy as np
import torch as th
from taxonomy_loss import TaxonomyLoss


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias   = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_Diffuse_model(nn.Module):
    """
    Combined model: Category Embedding + Taxonomy Loss

    Parameters
    ----------
    diffu        : DiffuRec instance
    args         : argparse.Namespace
    category_map : dict  item_id -> list[int] hoặc int
    category_num : int   số category unique
    """

    def __init__(self, diffu, args, category_map=None, category_num=0):
        super().__init__()
        self.emb_dim  = args.hidden_size
        self.item_num = args.item_num + 1
        self.max_len  = args.max_len

        # ── Item embedding ────────────────────────────────────────────────
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)

        # ── Category Embedding (phương pháp 1) ───────────────────────────
        self.use_category = (category_map is not None) and (category_num > 0)
        self.cat_alpha    = getattr(args, 'cat_alpha', 0.05)

        if self.use_category:
            self.category_embeddings = nn.Embedding(
                category_num + 1, self.emb_dim, padding_idx=0
            )
            # Precompute lookup table — vectorized, GPU-friendly
            max_cats       = 3
            item_cat_table = torch.zeros(self.item_num, max_cats, dtype=torch.long)
            for item_id, cat_ids in category_map.items():
                if item_id >= self.item_num:
                    continue
                if not isinstance(cat_ids, list):
                    cat_ids = [cat_ids]
                for j, c in enumerate(cat_ids[:max_cats]):
                    item_cat_table[item_id, j] = int(c)
            self.register_buffer('item_cat_table', item_cat_table)
            print(f"[CatEmb] ✅ | num_categories={category_num} | "
                  f"cat_alpha={self.cat_alpha}")
        else:
            print("[CatEmb] Disabled")

        # ── Taxonomy Loss (phương pháp 2) ────────────────────────────────
        multi_label = False
        if category_map is not None:
            sample_val  = next(iter(category_map.values()))
            multi_label = isinstance(sample_val, list) and len(sample_val) > 1

        self.taxonomy_loss_fn = TaxonomyLoss(
            item_num     = self.item_num,
            category_map = category_map,
            hidden_size  = args.hidden_size,
            alpha        = getattr(args, 'alpha_taxonomy', 0.1),
            beta         = getattr(args, 'beta_taxonomy',  0.6),
            margin       = getattr(args, 'margin_taxonomy', 0.8),
            loss_scale   = getattr(args, 'loss_scale',     3.0),
            device       = args.device,
            multi_label  = multi_label,
        )

        # ── Các layer chung ───────────────────────────────────────────────
        self.embed_dropout       = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm           = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout             = nn.Dropout(args.dropout)
        self.diffu               = diffu
        self.loss_ce             = nn.CrossEntropyLoss()
        self.loss_ce_rec         = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse            = nn.MSELoss()

    # ── Category embedding helpers ────────────────────────────────────────

    def _get_cat_emb(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (...,) → (..., H)"""
        cat_ids = self.item_cat_table[ids]           # (..., max_cats)
        cat_emb = self.category_embeddings(cat_ids)  # (..., max_cats, H)
        return cat_emb.mean(dim=-2)                  # (..., H)

    # ── Loss methods ──────────────────────────────────────────────────────

    def loss_diffu_ce(self, rep_diffu, labels):
        """L_CE : item prediction loss (gốc DiffuRec)"""
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_taxonomy_total(self, rep_diffu, labels):
        """L_taxonomy : Focal + Hard Triplet (từ TaxonomyLoss module)"""
        return self.taxonomy_loss_fn(
            rep_diffu       = rep_diffu,
            item_labels     = labels,
            item_embeddings = self.item_embeddings,
        )

    def loss_combined(self, rep_diffu, labels, warmup_weight: float = 1.0):
        """
        L_total = L_CE + warmup_weight * L_taxonomy

        warmup_weight: float trong [0, 1] — do trainer điều khiển
                       0.0 = chỉ train CE (warm-up giai đoạn đầu)
                       1.0 = full taxonomy loss
        """
        l_ce  = self.loss_diffu_ce(rep_diffu, labels)
        l_tax = self.taxonomy_loss_fn(rep_diffu, labels, self.item_embeddings)
        return l_ce + warmup_weight * l_tax, l_ce, l_tax

    @torch.no_grad()
    def category_accuracy(self, rep_diffu, labels) -> float:
        return self.taxonomy_loss_fn.category_accuracy(rep_diffu, labels)

    # ── Các method gốc ────────────────────────────────────────────────────

    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        return self.diffu(item_rep, tag_emb, mask_seq)

    def reverse(self, item_rep, noise_x_t, mask_seq):
        return self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq)

    def diffu_rep_pre(self, rep_diffu):
        return torch.matmul(rep_diffu, self.item_embeddings.weight.t())

    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))

    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight ** 2).sum(-1).view(-1, 1)
        rep_norm  = (rep_diffu ** 2).sum(-1).view(-1, 1)
        sim  = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        return -(torch.clamp(dist, 0.0, np.inf))

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, sequence, tag, train_flag=True):
        """
        sequence : (B, L)
        tag      : (B, 1)  — ground-truth item khi train, dummy khi infer
        """
        # Phương pháp 1: Category Embedding enrichment
        item_embeddings = self.item_embeddings(sequence)       # (B, L, H)
        if self.use_category:
            cat_emb = self._get_cat_emb(sequence)              # (B, L, H)
            item_embeddings = item_embeddings + self.cat_alpha * cat_emb

        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.LayerNorm(item_embeddings)
        mask_seq = (sequence > 0).float()

        if train_flag:
            tag_ids = tag.squeeze(-1)                          # (B,)
            tag_emb = self.item_embeddings(tag_ids)            # (B, H)
            if self.use_category:
                tag_emb = tag_emb + self.cat_alpha * self._get_cat_emb(tag_ids)

            rep_diffu, rep_item, weights, t = self.diffu_pre(
                item_embeddings, tag_emb, mask_seq
            )
            return None, rep_diffu, weights, t, None, None
        else:
            noise_x_t = th.randn_like(item_embeddings[:, -1, :])
            rep_diffu = self.reverse(item_embeddings, noise_x_t, mask_seq)
            return None, rep_diffu, None, None, None, None


def create_model_diffu(args):
    return DiffuRec(args)
