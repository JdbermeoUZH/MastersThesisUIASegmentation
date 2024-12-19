
import os

import numpy as np
from torch import Tensor, einsum
import torch


def write_summary(logdir, dice_scores_fg, dice_scores_classes_of_interest):
    """
    Write summary of dice scores to a file and print to console.

    Parameters
    ----------
    logdir : str
        Directory to save the summary file.
    dice_scores_fg : dict
        Dictionary of foreground dice scores.
    dice_scores_classes_of_interest : dict
        Dictionary of dice scores for classes of interest.

    Returns
    -------
    None
    """
    summary_file = os.path.join(logdir, "summary.txt")
    with open(summary_file, "w") as f:

        def write_and_print(text):
            print(text)
            f.write(text + "\n")

        def write_scores(scores, prefix=""):
            for name, values in scores.items():
                mean, std = np.mean(values), np.std(values)
                write_and_print(f"{prefix}{name} : {mean:.3f} +- {std:.5f}")

        write_and_print("Dataset level metrics\n")
        write_scores(dice_scores_fg)

        if len(dice_scores_classes_of_interest) > 0:
            write_and_print("\nClass level metrics\n")
            for cls, scores in dice_scores_classes_of_interest.items():
                write_and_print(f"Class {cls}")
                write_scores(scores, prefix="  ")

            write_and_print("\nClass average")
            avg_scores = {
                name: [
                    np.mean(cls_scores[name])
                    for cls_scores in dice_scores_classes_of_interest.values()
                ]
                for name in dice_scores_fg
            }
            write_scores(avg_scores, prefix="  ")


def norm_soft_size(probs_per_pix: Tensor, power:int) -> Tensor:
    """
    Normalize over each channel

    Assumes probs_per_pix is tensor with probabilites of shape (b,c,w,h)
    for each pixel in the image of shape (w,h).

    Probabilites are scaled by the maximum probability in each channel to 
    avoid numerical instability.

    """
    b, c, w, h = probs_per_pix.shape
    sl_sz = w*h
    amax = probs_per_pix.max(dim=1, keepdim=True)[0]+1e-10
    #amax = torch.cat(c*[amax], dim=1)
    resp = (torch.div(probs_per_pix,amax))**power
    ress = einsum("bcwh->bc", [resp]).type(torch.float32)
    ress_norm = ress/(torch.sum(ress,dim=1,keepdim=True)+1e-10)
    #print(torch.sum(ress,dim=1))
    
    return ress_norm.unsqueeze(2)