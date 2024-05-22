from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.feature import SIFTDescriptor
from kornia.core import  Tensor, concatenate, normalize
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.filters import spatial_gradient
from kornia.geometry.conversions import pi

from tta_uia_segmentation.src.utils.utils import assert_in, from_dict_or_default


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


def dice_score(mask_pred, mask_gt, soft=True, reduction='mean', bg_channel=0, k=1, smooth=0, epsilon=1e-10):
    """ 
    Assumes that mask_pred and mask_gt are one-hot encoded.
    
    Parameters
    ----------
    mask_pred : torch.Tensor
        Predicted masks.
    
    mask_gt : torch.Tensor

    soft : bool
        If True, mask_pred is assumed to be soft and will be converted to one-hot.
    
    reduction : str
        Reduction method.
    
    bg_channel : int
        Background channel.
    
    k : int
        Exponent.
    
    smooth : float
        Smoothing factor.
        
    epsilon : float
        Small value to avoid division by zero. 
        
    """

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
    dices = (2 * tp + smooth) / (tp_plus_fp + tp_plus_fn + smooth + epsilon)

    assert_in(reduction, 'reduction', ['none', 'mean', 'sum'])

    fg_mask = (torch.arange(mask_pred.shape[1]) != bg_channel)
    if reduction == 'none':
        return dices, dices[:, fg_mask, ...]
    elif reduction == 'mean':
        return dices.nanmean(), dices[:, fg_mask, ...].nanmean()
    elif reduction == 'sum':
        return dices.nansum(), dices[:, fg_mask, ...].nansum()


class DiceLoss(nn.Module):
    def __init__(self, smooth = 0, epsilon=1e-10):
        """
        Dice loss.
        
        Attributes
        ----------
        smooth : float
            Smoothing factor. We keep it at 0 so that the loss is maximum numerator and denominator are 0.
            (Dice score is 1 when both numerator and denominator are 0, so loss is 1)
            
        epsilon : float
            Small value to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth
        self.epsilon = epsilon

    def forward(self, mask_pred, mask_gt, **kwargs):
            
        dice, _ = dice_score(
            mask_pred, mask_gt, 
            soft=True,
            smooth=self.smooth, epsilon=self.epsilon, 
            **kwargs
            )
        loss = 1 - dice

        return loss


class SIFTDescriptor(SIFTDescriptor):
    def __init__(
        self,
        patch_size: int = 41,
        num_ang_bins: int = 8,
        num_spatial_bins: int = 4,
        rootsift: bool = True,
        clipval: float = 0.2,
        use_rsq_grads: bool = False
        ): 
        super().__init__(patch_size, num_ang_bins, num_spatial_bins, rootsift, clipval)
        self.use_rsq_grads = use_rsq_grads
    
    def forward(self, input: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(input, ["B", "1", f"{self.patch_size}", f"{self.patch_size}"])
        B: int = input.shape[0]
        self.pk = self.pk.to(input.dtype).to(input.device)

        grads = spatial_gradient(input, "diff")
        # unpack the edges
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]
        
        if self.use_rsq_grads:
            gx = torch.sqrt(gx**2)
            gy = torch.sqrt(gy**2)

        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        mag = mag * self.gk.expand_as(mag).type_as(mag).to(mag.device)
        o_big = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_ = torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big = bo0_big_ % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag

        ang_bins = concatenate(
            [
                self.pk((bo0_big == i).to(input.dtype) * wo0_big + (bo1_big == i).to(input.dtype) * wo1_big)
                for i in range(0, self.num_ang_bins)
            ],
            1,
        )
        ang_bins = ang_bins.view(B, -1)
        ang_bins = normalize(ang_bins, p=2)
        ang_bins = torch.clamp(ang_bins, 0.0, float(self.clipval))
        ang_bins = normalize(ang_bins, p=2)
        if self.rootsift:
            ang_bins = torch.sqrt(normalize(ang_bins, p=1) + self.eps)
        return ang_bins
    
    def describe_image_sift(self, image):
        image = image.unfold(-2, self.patch_size, self.patch_size)\
            .unfold(-2, self.patch_size, self.patch_size)
        descs = self(image.reshape(-1, 1, self.patch_size, self.patch_size))
        return descs
    

class SIFDescriptorLoss(nn.Module):
    def __init__(self, patch_size=256, num_ang_bins=8, num_spatial_bins=16, use_rsq_grads=True, **kwargs):
        super().__init__()
        self.sift = SIFTDescriptor(patch_size=patch_size, num_ang_bins=num_ang_bins, num_spatial_bins=num_spatial_bins, 
                                   use_rsq_grads=use_rsq_grads, **kwargs)

    def forward(self, img_1, img_2):
        descs_1 = self.sift.describe_image_sift(img_1)
        descs_2 = self.sift.describe_image_sift(img_2)

        loss = F.mse_loss(descs_1, descs_2)

        return loss
    
    
def zncc(x1, x2, return_squared = True):
    """
    Computes the Zero Mean Normalized Cross Correlation (ZNCC) between two images.
    
    Args:
        x1 (torch.Tensor): The first image tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the image respectively.
        x2 (torch.Tensor): The second image tensor of the same shape as x1.
        
    Returns:
        torch.Tensor: A tensor of shape (B) containing the ZNCC values for each pair of corresponding images in x1 and x2.
    """
    # Compute means and standard deviations along channel dimension
    mu_x1 = x1.mean((2,3), keepdim=True)
    mu_x2 = x2.mean((2,3), keepdim=True)
        
    # Subtract means and divide by standard deviations
    x1_zm = (x1 - mu_x1)
    x2_zm = (x2 - mu_x2) 

    # Compute dot product and normalize by square root of sum of squares
    dp = torch.sum(x1_zm * x2_zm, dim=(1,2,3))
    norm = torch.sqrt(torch.sum(x1_zm**2, dim=(1,2,3))) * torch.sqrt(torch.sum(x2_zm**2, dim=(1,2,3)))
    zncc = dp / (norm + 1e-8)
    
    if return_squared:
        zncc = zncc**2

    return zncc


class ZNCCLoss(nn.Module):
    def __init__(self, return_squared=True):
        super().__init__()
        self.return_squared = return_squared

    def forward(self, x1, x2):
        zncc_loss = zncc(x1, x2, return_squared=self.return_squared)
        return 1 - zncc_loss.mean()


class MutualInformationLoss(nn.Module):
    """ 
    Compute the Mutual Information between two images using a kernel density estimation approach.
    
    From: https://github.com/connorlee77/pytorch-mutual-information
    """
    
    def __init__(self, sigma=0.1, num_bins=256, normalize=True):
        super().__init__()

        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)


    def marginalPdf(self, values):

        residuals = values - self.bins.unsqueeze(0).unsqueeze(0).to(values.device)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        
        return pdf, kernel_values


    def jointPdf(self, kernel_values1, kernel_values2):

        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
        normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf


    def getMutualInformation(self, input1, input2):
        '''
            input1: B, C, H, W
            input2: B, C, H, W

            return: scalar
        '''

        # Torch tensors for images between (0, 1)
        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W = input1.shape
        assert((input1.shape == input2.shape))

        x1 = input1.view(B, H*W, C)
        x2 = input2.view(B, H*W, C)
        
        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

        mutual_information = H_x1 + H_x2 - H_x1x2
        
        if self.normalize:
            mutual_information = mutual_information / (H_x1+H_x2)

        return mutual_information

    def forward(self, input1, input2):
        '''
            input1: B, C, H, W
            input2: B, C, H, W

            return: scalar
        '''
            
        return 1 - self.getMutualInformation(input1, input2).mean()


class BasicGradientComparissonLoss(nn.Module):
    def __init__(self, use_sq_grads=True, mode='diff'):
        super().__init__()
        self.use_sq_grads = use_sq_grads
        self.mode = mode

    def forward(self, x1, x2):
        grad_x1 = spatial_gradient(x1, self.mode)
        grad_x2 = spatial_gradient(x2, self.mode)

        if self.use_sq_grads:
            grad_x1 = grad_x1**2
            grad_x2 = grad_x2**2
    
        return F.mse_loss(grad_x1, grad_x2)



class DescriptorRegularizationLoss(nn.Module):
    """
    TODO
     - Add term that penalizes low vaiances in the input image
     - add method to crop around the center of the image a patch that is 75% the size of the image
    """
    def __init__(self, type: Literal['sq_grad', 'rsq_sift', 'sift', 'zncc', 'mi'], **kwargs):
        super().__init__()
        if type == 'rsq_sift':
            self.loss_fn = SIFDescriptorLoss(
                patch_size=from_dict_or_default(kwargs, 'patch_size', 256),
                num_ang_bins=from_dict_or_default(kwargs, 'num_ang_bins', 8),
                num_spatial_bins=from_dict_or_default(kwargs, 'num_spatial_bins', 16),
                use_rsq_grads=from_dict_or_default(kwargs, 'use_rsq_grads', True)   
                )

        if type == 'sift':
            self.loss_fn = SIFDescriptorLoss(
                patch_size=from_dict_or_default(kwargs, 'patch_size', 256),
                num_ang_bins=from_dict_or_default(kwargs, 'num_ang_bins', 8),
                num_spatial_bins=from_dict_or_default(kwargs, 'num_spatial_bins', 16),
                use_rsq_grads=from_dict_or_default(kwargs, 'use_rsq_grads', False)   
                )
        
        elif type == 'zncc':
            self.loss_fn = ZNCCLoss(
                return_squared=from_dict_or_default(kwargs, 'return_squared', True)
                )
            
        elif type == 'mi':
            self.loss_fn = MutualInformationLoss(
                sigma=from_dict_or_default(kwargs, 'sigma', 0.1),
                num_bins=from_dict_or_default(kwargs, 'num_bins', 256),
                normalize=from_dict_or_default(kwargs, 'normalize', True)
                )        
        elif type == 'sq_grad':
            self.loss_fn = BasicGradientComparissonLoss(
                use_sq_grads=from_dict_or_default(kwargs, 'use_sq_grads', True),
                mode=from_dict_or_default(kwargs, 'mode', 'diff')
                )
        else:
            raise ValueError(f'Unknown descriptor type: {type}')

    def forward(self, x1, x2):
        return self.loss_fn(x1, x2)
        
        
if __name__ == '__main__':
    test_pred = torch.randint(0, 5, (1, 1, 64, 64, 64))
    test_gt = torch.randint(0, 5, (1, 1, 64, 64, 64))
    
    