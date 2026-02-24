"""
Specialised Loss Functions for Student Group Selection
=======================================================

All three losses are designed for the same task:
  Input : logit vector (n_students,) — RAW linear output, no sigmoid
  Target: binary vector (n_students,) — 1 for selected, 0 otherwise
  Exactly k=6 ones per target row.

Why NOT standard BCELoss?
--------------------------
BCELoss treats each of the 55 students as an INDEPENDENT binary variable.
It has two fundamental weaknesses for this task:

  1. Class imbalance: 49 negatives vs 6 positives per day (8:1 ratio).
     The gradient is dominated by easy negatives.

  2. Independence assumption: it does not care whether the TOP-6 predicted
     students match the actual 6 — only whether EACH individual probability
     is high for selected and low for non-selected.

The three losses below address these weaknesses differently:

  ┌───────────────┬────────────────────────────────────────────────────┐
  │ Loss          │ Core Idea                                          │
  ├───────────────┼────────────────────────────────────────────────────┤
  │ RankingLoss   │ Treat selection as a ranking problem. Softmax over │
  │ (ListNet)     │ ALL 55 logits → cross-entropy with 1/k for each    │
  │               │ selected student. Forces the MODEL to produce      │
  │               │ logits where selected > non-selected globally.     │
  ├───────────────┼────────────────────────────────────────────────────┤
  │ TopKLoss      │ Focus on the hardest negative examples. BCE only   │
  │               │ on: (all k positives) + (k highest-confidence      │
  │               │ wrong predictions). Ignores easy negatives.        │
  │               │ Similar to Online Hard Example Mining (OHEM).      │
  ├───────────────┼────────────────────────────────────────────────────┤
  │ FocalLoss     │ Down-weight easy predictions via (1-p_t)^gamma.    │
  │               │ Students the model already correctly assigns low   │
  │               │ probability contribute nearly zero gradient.       │
  │               │ Forces learning from uncertain / misclassified     │
  │               │ students. gamma=2 is standard from Lin et al.      │
  └───────────────┴────────────────────────────────────────────────────┘

All losses operate on LOGITS (pre-sigmoid) for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
#  1. Ranking Loss (ListNet / Plackett-Luce)
# ──────────────────────────────────────────────────────────────────────────

class RankingLoss(nn.Module):
    """
    ListNet-style listwise ranking loss for multi-label selection.

    Motivation:
        We want the model to RANK the 6 selected students above the 49 others.
        BCELoss doesn't directly optimise for ranking — it optimises each
        student independently. RankingLoss uses softmax over all logits,
        making every student's gradient depend on all others.

    Mathematics:
        target_dist[i] = 1/k  if student i was selected
                       = 0     otherwise

        loss = -sum_i  target_dist[i] * log(softmax(logits)[i])
             = -mean_i∈selected  log(softmax(logits)[i])

        This is equivalent to the Plackett-Luce model: maximise the
        log-likelihood of the selection under a multinomial with
        probabilities proportional to exp(logits).

    Effect on training:
        - Increasing one selected student's logit DECREASES all others'
          softmax values → tighter competition at the boundary.
        - Students around rank 5-10 receive the strongest gradient signal.

    Args:
        temperature: Sharpens (T<1) or flattens (T>1) the target distribution.
                     T=1.0 means uniform probability over selected students.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, n_students) raw linear outputs.
            targets: (batch, n_students) binary {0, 1}, sum=k per row.

        Returns:
            Scalar loss.
        """
        # ── Target distribution ─────────────────────────────────────────
        k = targets.sum(dim=-1, keepdim=True).clamp(min=1.0)

        if self.temperature != 1.0:
            # Soften the target: smooth the 1/k mass slightly
            soft_pos = 1.0 / self.temperature
            target_dist = targets * soft_pos
            target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        else:
            target_dist = targets / k  # uniform over selected students

        # ── Log-softmax of logits ────────────────────────────────────────
        log_pred = F.log_softmax(logits, dim=-1)

        # ── Negative log-likelihood ──────────────────────────────────────
        loss = -(target_dist * log_pred).sum(dim=-1).mean()
        return loss


# ──────────────────────────────────────────────────────────────────────────
#  2. Top-K Loss (Online Hard Example Mining)
# ──────────────────────────────────────────────────────────────────────────

class TopKLoss(nn.Module):
    """
    Top-K focused binary cross-entropy with hard negative mining.

    Motivation:
        With 55 students and only 6 selected, 49 students are "easy" negatives
        after a few training epochs (model correctly gives them low probability).
        These easy negatives dominate the BCE gradient without improving ranking.

        TopKLoss focuses each training step on the HARDEST examples:
          Positives  : all k=6 actually selected students (always included)
          Hard Neg.  : the k students with the HIGHEST predicted probability
                       AMONG the non-selected ones (most likely to be rank-7/8)

        Together these 2k students form the focus set. Only BCE on this set
        is computed. The gradient is concentrated where it matters most.

    Connection to OHEM:
        This is Online Hard Example Mining applied to per-student binary
        classification. The "mining" is done per-day, per-sample.

    Args:
        k:           Number of positives = number of hard negatives to mine.
        margin:      Optional margin to add to hard-negative logits before
                     loss computation (encourages a larger decision boundary).
    """

    def __init__(self, k: int = 6, margin: float = 0.0):
        super().__init__()
        self.k      = k
        self.margin = margin

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, n_students) raw linear outputs.
            targets: (batch, n_students) binary {0, 1}.

        Returns:
            Scalar loss.
        """
        batch_size, n_students = logits.shape
        losses = []

        # Detach sigmoid for mining (no gradient through mining step)
        with torch.no_grad():
            probs_det = torch.sigmoid(logits)

        for b in range(batch_size):
            pos_mask = targets[b] > 0.5          # (n_students,) bool

            # ── Hard negative mining ───────────────────────────────────
            neg_scores = probs_det[b].clone()
            neg_scores[pos_mask] = -1.0           # exclude true positives
            _, hard_idx = neg_scores.topk(self.k)

            neg_mask = torch.zeros(n_students, dtype=torch.bool,
                                   device=logits.device)
            neg_mask[hard_idx] = True

            # ── Focus set = positives + hard negatives ─────────────────
            focus = pos_mask | neg_mask

            focus_logits  = logits[b][focus]
            focus_targets = targets[b][focus]

            if self.margin > 0:
                # Push hard negatives further from the boundary
                neg_in_focus = (~pos_mask)[focus]
                focus_logits = focus_logits.clone()
                focus_logits[neg_in_focus] = (focus_logits[neg_in_focus]
                                              + self.margin)

            loss = F.binary_cross_entropy_with_logits(
                focus_logits, focus_targets)
            losses.append(loss)

        return torch.stack(losses).mean()


# ──────────────────────────────────────────────────────────────────────────
#  3. Focal Loss (Lin et al. 2017)
# ──────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for dense binary prediction with class imbalance.

    Originally proposed for object detection (RetinaNet, Lin et al. 2017),
    adapted here for multi-label student selection prediction.

    Mathematics:
        FL(logit_i, y_i) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        where  p_t  = sigmoid(logit_i)   if y_i = 1
                    = 1 - sigmoid(logit_i) if y_i = 0

    Key properties:
        gamma=0          → standard weighted BCE
        gamma=1          → linear down-weighting of easy examples
        gamma=2 (default)→ easy examples (p_t > 0.9) lose ~99% of their weight

        With 6 positives vs 49 negatives per day:
          alpha=0.25 (positives) vs 0.75 (negatives) weights the positive
          class MORE to counteract the 8:1 imbalance.

    Args:
        alpha:  Weighting factor for the positive class (selected students).
                Recommended: 0.25–0.5 for 8:1 imbalance.
        gamma:  Focusing exponent. Higher = more focus on hard examples.
                0 = standard BCE, 2 = standard Focal, 5 = extreme focus.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, n_students) raw linear outputs.
            targets: (batch, n_students) binary {0, 1}.

        Returns:
            Scalar loss.
        """
        # ── Numerically stable BCE via log-sigmoid ───────────────────────
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')

        # ── p_t: probability of being correct ───────────────────────────
        probs = torch.sigmoid(logits)
        p_t   = torch.where(targets > 0.5, probs, 1.0 - probs)

        # ── Focal modulation ─────────────────────────────────────────────
        focal_weight = (1.0 - p_t) ** self.gamma

        # ── Alpha per-class weighting ────────────────────────────────────
        alpha_t = torch.where(targets > 0.5,
                               torch.full_like(targets, self.alpha),
                               torch.full_like(targets, 1.0 - self.alpha))

        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────
#  Convenience: BCE baseline (same interface)
# ──────────────────────────────────────────────────────────────────────────

class BCELoss(nn.Module):
    """
    Standard BCE on logits — identical to nn.BCEWithLogitsLoss.
    Included for a clean apples-to-apples comparison.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets)
