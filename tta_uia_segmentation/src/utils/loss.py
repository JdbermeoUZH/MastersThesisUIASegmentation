import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import assert_in


def class_to_onehot(class_image, n_classes=-1, class_dim=1):
    """
    class_dim: dimension where classes are added (0 if class_image is an image and 1 if class_image is a batch)
    """
    class_image = class_image.long()
    one_hot = F.one_hot(class_image, n_classes).byte()
    one_hot = one_hot.squeeze(class_dim).movedim(-1, class_dim)

    return one_hot


def onehot_to_class(onehot, class_dim=1, keepdim=True):
    return onehot.argmax(dim=class_dim, keepdim=keepdim)


def dice_score(mask_pred, mask_gt, soft=True, reduction='mean', bg_channel=0, k=1, epsilon=0):

    if not soft:
        n_classes = mask_pred.shape[1]
        mask_pred = class_to_onehot(onehot_to_class(mask_pred), n_classes)

    N, C = mask_pred.shape[0:2]
    mask_pred = mask_pred.reshape(N, C, -1)
    mask_gt = mask_gt.reshape(N, C, -1)

    assert mask_pred.shape == mask_gt.shape

    tp = torch.sum(mask_gt * mask_pred, dim=-1)
    tp_plus_fp = torch.sum(mask_pred ** k, dim=-1)
    tp_plus_fn = torch.sum(mask_gt ** k, dim=-1)
    dices = (2 * tp + epsilon) / (tp_plus_fp + tp_plus_fn + epsilon)

    assert_in(reduction, 'reduction', ['none', 'mean', 'sum'])

    fg_mask = (torch.arange(mask_pred.shape[1]) != bg_channel)
    if reduction == 'none':
        return dices, dices[:, fg_mask, ...]
    elif reduction == 'mean':
        return dices.nanmean(), dices[:, fg_mask, ...].nanmean()
    elif reduction == 'sum':
        return dices.nansum(), dices[:, fg_mask, ...].nansum()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_pred, mask_gt, **kwargs):

        dice, dice_fg = dice_score(mask_pred, mask_gt, **kwargs)
        loss = 1 - dice

        return loss


if __name__ == '__main__':
    test_pred = torch.randint(0, 5, (1, 1, 64, 64, 64))
    test_gt = torch.randint(0, 5, (1, 1, 64, 64, 64))
    
    