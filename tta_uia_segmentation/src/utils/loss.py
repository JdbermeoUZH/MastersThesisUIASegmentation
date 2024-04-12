from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.feature import SIFTDescriptor

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
    

class SIFTDescriptor(SIFTDescriptor):
    def describe_image_sift(self, image):
        image = image.unfold(-2, self.patch_size, self.patch_size)\
            .unfold(-2, self.patch_size, self.patch_size)
        descs = self(image.reshape(-1, 1, self.patch_size, self.patch_size))
        return descs
    

class SIFDescriptorLoss(nn.Module):
    def __init__(self, patch_size=256, num_ang_bins=8, num_spatial_bins=16, **kwargs):
        super().__init__()
        self.sift = SIFTDescriptor(patch_size=patch_size, num_ang_bins=num_ang_bins, num_spatial_bins=num_spatial_bins, **kwargs)

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


class DescriptorRegularizationLoss(nn.Module):
    """
    TODO
     - Add term that penalizes low vaiances in the input image
     - add method to crop around the center of the image a patch that is 75% the size of the image
    """
    def __init__(self, type: Literal['sift', 'zncc', 'mi'], **kwargs):
        super().__init__()
        if type == 'sift':
            self.loss_fn = SIFDescriptorLoss(
                patch_size=from_dict_or_default(kwargs, 'patch_size', 256),
                num_ang_bins=from_dict_or_default(kwargs, 'num_ang_bins', 8),
                num_spatial_bins=from_dict_or_default(kwargs, 'num_spatial_bins', 16)
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
        else:
            raise ValueError(f'Unknown descriptor type: {type}')

    def forward(self, x1, x2):
        return self.loss_fn(x1, x2)
        
        
if __name__ == '__main__':
    test_pred = torch.randint(0, 5, (1, 1, 64, 64, 64))
    test_gt = torch.randint(0, 5, (1, 1, 64, 64, 64))
    
    