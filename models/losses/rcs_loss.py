"""Region-level Contrastive Supervision (RCS) loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection.
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def rcs_loss(pred_masks, gt_masks, num_masks, temperature=0.1, margin=0.3, ignore_index=255):
    """
    Region-level Contrastive Supervision on upsampled predictions.
    
    Args:
        pred_masks: [B, C, H, W] - Predicted logits (already upsampled)
        gt_masks: [B, H, W] - Ground truth segmentation masks
        num_masks: Number of masks for normalization
        temperature: Temperature for similarity scaling
        margin: Margin for triplet-style loss
        ignore_index: Index to ignore in ground truth
        
    Returns:
        loss: Scalar loss value
    """
    B, C, H, W = pred_masks.shape
    device = pred_masks.device
    
    logits = pred_masks.view(B, C, -1)
    
    gt = gt_masks.view(B, -1)
    
    valid = (gt != ignore_index) & (gt >= 0) & (gt < C)
    num_valid = valid.sum()
    
    if num_valid == 0:
        return pred_masks.new_zeros(1)[0]
    
    gt_safe = gt.clamp(0, C - 1).unsqueeze(1)
    pos_logits = logits.gather(1, gt_safe).squeeze(1)
    
    max_logits, max_idx = logits.max(dim=1)
    
    is_correct = (max_idx == gt_safe.squeeze(1))
    
    diff = (max_logits - pos_logits) / temperature + margin
    loss_per_pixel = F.relu(diff) * (~is_correct).float() * valid.float()
    
    loss = loss_per_pixel.sum() / (num_valid.float() + 1e-6)
    
    return loss


class SegPlusCriterionWithRCS(nn.Module):
    """SegPlusCriterion with an added RCS loss term."""

    def __init__(self, num_classes, weight_dict, losses, eos_coef=0.1,
                 rcs_temperature=0.1, rcs_margin=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.rcs_temperature = rcs_temperature
        self.rcs_margin = rcs_margin
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: focal, dice, and RCS loss."""
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"]
        target_masks = self._get_target_mask_binary_cross_entropy(src_masks, targets)

        bs, n_cls, H, W = target_masks.size()
        _, _, H_, W_ = src_masks.size()
        src_masks_focal = src_masks.reshape(bs*n_cls, H_, W_)
        target_masks_focal = target_masks.reshape(bs*n_cls, H, W)
        
        src_masks_focal = F.interpolate(
            src_masks_focal[:, None], size=target_masks_focal.shape[-2:], 
            mode="bilinear", align_corners=False
        )
        src_masks_focal = src_masks_focal[:, 0].flatten(1)
        target_masks_focal = target_masks_focal.flatten(1)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks_dice = outputs["pred_masks"]
        if src_masks_dice.dim() != 4:
            return {"no_loss": 0}
        src_masks_dice = src_masks_dice[src_idx]
        masks_dice = [t["target_masks"] for t in targets]
        target_masks_dice, valid = nested_tensor_from_tensor_list(masks_dice).decompose()
        target_masks_dice = target_masks_dice.to(src_masks_dice)
        target_masks_dice = target_masks_dice[tgt_idx]

        src_masks_dice = F.interpolate(
            src_masks_dice[:, None], size=target_masks_dice.shape[-2:], 
            mode="bilinear", align_corners=False
        )
        src_masks_dice = src_masks_dice[:, 0].flatten(1)
        target_masks_dice = target_masks_dice.flatten(1)
        target_masks_dice = target_masks_dice.view(src_masks_dice.shape)

        pred_for_rcs = F.interpolate(
            outputs["pred_masks"], size=(H, W), 
            mode="bilinear", align_corners=False
        )
        gt_for_rcs = torch.stack([t["masks"] for t in targets], dim=0)
        
        loss_rcs = rcs_loss(
            pred_for_rcs, gt_for_rcs, num_masks,
            temperature=self.rcs_temperature,
            margin=self.rcs_margin
        )

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks_focal, target_masks_focal, num_masks),
            "loss_dice": dice_loss(src_masks_dice, target_masks_dice, num_masks),
            "loss_rcs": loss_rcs,
        }
        return losses

    def _get_target_mask_binary_cross_entropy(self, out_masks, targets):
        B, C = out_masks.size()[:2]
        H, W = targets[0]['masks'].size()
        target_masks_o = torch.zeros(B, C, H*W).to(out_masks.device) 
        for i, target in enumerate(targets):
            mask = target['masks'].long().reshape(-1)
            idx = torch.arange(0, H*W, 1).long().to(out_masks.device) 
            mask_o = mask[mask!=255]
            idx = idx[mask!=255]
            target_masks_o[i, mask_o, idx] = 1
        return target_masks_o.reshape(B, C, H, W)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        labels = [x['labels'] for x in targets]
        indices_new = []
        for label in labels:
            t = torch.arange(len(label))
            indices_new.append([label, t])
        indices = indices_new
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_loss = ["masks"]
                for loss in aux_loss:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


@LOSSES.register_module()
class SegLossPlusRCS(nn.Module):
    """SegLossPlus with RCS loss."""
    def __init__(self,
                 num_classes,
                 dec_layers,
                 mask_weight=20.0,
                 dice_weight=1.0,
                 rcs_weight=0.2,
                 loss_weight=1.0,
                 rcs_temperature=0.1,
                 rcs_margin=0.3,
                 use_hard_negative=True,
                 use_point=False):
        super(SegLossPlusRCS, self).__init__()
        
        weight_dict = {
            "loss_mask": mask_weight, 
            "loss_dice": dice_weight,
            "loss_rcs": rcs_weight
        }
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        self.criterion = SegPlusCriterionWithRCS(
            num_classes,
            weight_dict=weight_dict,
            losses=["masks"],
            rcs_temperature=rcs_temperature,
            rcs_margin=rcs_margin,
        )

        self.loss_weight = loss_weight
        self.ignore_index = 255

    def forward(self,
                outputs,
                label,
                text_embeddings=None,
                ignore_index=255,
                ):
        """Forward function - same signature as original SegLossPlus."""
        
        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                losses.pop(k)

        return losses

    def prepare_targets(self, targets):
        """Same as original SegLossPlus."""
        new_targets = []
        for targets_per_image in targets:
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            new_targets.append(
                {
                    "labels": gt_cls,
                    "target_masks": masks,
                    "masks": targets_per_image,
                }
            )
        return new_targets


