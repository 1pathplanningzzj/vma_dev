# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from mmdet.models.builder import LOSSES
from mmdet.models import weighted_loss
import mmcv
import torch.nn.functional as F
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import functools
import math
import threading

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

@mmcv.jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_dir
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            # loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum()
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

@mmcv.jit(derivate=True, coderize=True)
def custom_weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_order, num_pts, num_coords
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum((1,2,3))
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def custom_weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def custom_weighted_dir_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_smooth_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss =smooth_l1_loss(pred,target, reduction='none')
    # import pdb;pdb.set_trace()
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    # import pdb;pdb.set_trace()
    return loss

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    """ Dir cosine similiarity loss
    pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
    target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0,1), target.flatten(0,1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss

@LOSSES.register_module()
class OrderedPtsSmoothL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * ordered_pts_smooth_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@LOSSES.register_module()
class PtsDirCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_dir



@LOSSES.register_module()
class PtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@LOSSES.register_module()
class OrderedPtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * ordered_pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@MATCH_COST.register_module()
class OrderedPtsSmoothL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1).unsqueeze(1).repeat(1,num_gts*num_orders,1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1).unsqueeze(0).repeat(bbox_pred.size(0),1,1)
        # import pdb;pdb.set_trace()
        bbox_cost = smooth_l1_loss(bbox_pred, gt_bboxes, reduction='none').sum(-1)
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class PtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.view(num_gts,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class OrderedPtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class MyChamferDistanceCost:
    def __init__(self, loss_src_weight=1., loss_dst_weight=1.):
        # assert mode in ['smooth_l1', 'l1', 'l2']
        # self.mode = mode
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def __call__(self, src, dst,src_weight=1.0,dst_weight=1.0,):
        """
        pred_pts (Tensor): normed coordinate(x,y), shape (num_q, num_pts_M, 2)
        gt_pts (Tensor): normed coordinate(x,y), shape (num_gt, num_pts_N, 2)
        """
        # criterion_mode = self.mode
        # if criterion_mode == 'smooth_l1':
        #     criterion = smooth_l1_loss
        # elif criterion_mode == 'l1':
        #     criterion = l1_loss
        # elif criterion_mode == 'l2':
        #     criterion = mse_loss
        # else:
        #     raise NotImplementedError
        # import pdb;pdb.set_trace()
        src_expand = src.unsqueeze(1).repeat(1,dst.shape[0],1,1)
        dst_expand = dst.unsqueeze(0).repeat(src.shape[0],1,1,1)
        # src_expand = src.unsqueeze(2).unsqueeze(1).repeat(1,dst.shape[0], 1, dst.shape[1], 1)
        # dst_expand = dst.unsqueeze(1).unsqueeze(0).repeat(src.shape[0],1, src.shape[1], 1, 1)
        distance = torch.cdist(src_expand, dst_expand)
        src2dst_distance = torch.min(distance, dim=3)[0]  # (num_q, num_gt, num_pts_N)
        dst2src_distance = torch.min(distance, dim=2)[0]  # (num_q, num_gt, num_pts_M)
        loss_src = (src2dst_distance * src_weight).mean(-1)
        loss_dst = (dst2src_distance * dst_weight).mean(-1)
        loss = loss_src*self.loss_src_weight + loss_dst * self.loss_dst_weight
        return loss

@mmcv.jit(derivate=True, coderize=True)
def chamfer_distance(src,
                     dst,
                     src_weight=1.0,
                     dst_weight=1.0,
                    #  criterion_mode='l1',
                     reduction='mean',
                     avg_factor=None):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    # if criterion_mode == 'smooth_l1':
    #     criterion = smooth_l1_loss
    # elif criterion_mode == 'l1':
    #     criterion = l1_loss
    # elif criterion_mode == 'l2':
    #     criterion = mse_loss
    # else:
    #     raise NotImplementedError

    # src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    # dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # import pdb;pdb.set_trace()
    distance = torch.cdist(src, dst)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)
    # import pdb;pdb.set_trace()
    #TODO this may be wrong for misaligned src_weight, now[N,fixed_num]
    # should be [N], then view
    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)
    if avg_factor is None:
        reduction_enum = F._Reduction.get_enum(reduction)
        if reduction_enum == 0:
            raise ValueError('MyCDLoss can not be used with reduction=`none`')
        elif reduction_enum == 1:
            loss_src = loss_src.mean(-1).mean()
            loss_dst = loss_dst.mean(-1).mean()
        elif reduction_enum == 2:
            loss_src = loss_src.mean(-1).sum()
            loss_dst = loss_dst.mean(-1).sum()
        else:
            raise NotImplementedError
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss_src = loss_src.mean(-1).sum() / (avg_factor + eps)
            loss_dst = loss_dst.mean(-1).sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss_src, loss_dst, indices1, indices2


@LOSSES.register_module()
class MyChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_src_weight (float): Weight of loss_source.
        loss_dst_weight (float): Weight of loss_target.
    """

    def __init__(self,
                #  mode='l1',
                 reduction='mean',
                 loss_src_weight=1.0,
                 loss_dst_weight=1.0):
        super(MyChamferDistance, self).__init__()

        # assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        # self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def forward(self,
                source,
                target,
                src_weight=1.0,
                dst_weight=1.0,
                avg_factor=None,
                reduction_override=None,
                return_indices=False,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            source (torch.Tensor): Source set with shape [B, N, C] to
                calculate Chamfer Distance.
            target (torch.Tensor): Destination set with shape [B, M, C] to
                calculate Chamfer Distance.
            src_weight (torch.Tensor | float, optional):
                Weight of source loss. Defaults to 1.0.
            dst_weight (torch.Tensor | float, optional):
                Weight of destination loss. Defaults to 1.0.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.
            return_indices (bool, optional): Whether to return indices.
                Defaults to False.

        Returns:
            tuple[torch.Tensor]: If ``return_indices=True``, return losses of \
                source and target with their corresponding indices in the \
                order of ``(loss_source, loss_target, indices1, indices2)``. \
                If ``return_indices=False``, return \
                ``(loss_source, loss_target)``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_source, loss_target, indices1, indices2 = chamfer_distance(
            source, target, src_weight, dst_weight, reduction,
            avg_factor=avg_factor)

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight

        loss_pts = loss_source + loss_target

        if return_indices:
            return loss_pts, indices1, indices2
        else:
            return loss_pts

def getDist_P2L(pred_point, gt_former, gt_current, gt_latter):
    # import pdb;pdb.set_trace()
    if gt_former.equal(gt_current):
        distance1 = math.sqrt(math.pow(pred_point[0]-gt_current[0], 2) + math.pow(pred_point[1]-gt_current[1],2))
    else:
        A=gt_former[1]-gt_current[1]
        B=gt_former[0]-gt_current[0]
        C=gt_former[0]*gt_current[1]-gt_former[1]*gt_current[0]
        distance1=(A*pred_point[0]+B*pred_point[1]+C)/math.sqrt(A*A+B*B)
    if gt_latter.equal(gt_current):
        distance2 = math.sqrt(math.pow(pred_point[0]-gt_current[0], 2) + math.pow(pred_point[1]-gt_current[1],2))
    else:
        A=gt_latter[1]-gt_current[1]
        B=gt_latter[0]-gt_current[0]
        C=gt_latter[0]*gt_current[1]-gt_latter[1]*gt_current[0]
        distance2=(A*pred_point[0]+B*pred_point[1]+C)/math.sqrt(A*A+B*B)
    
    return torch.Tensor([distance1 / 2.0, distance2 / 2.0 ])

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_2_lines_loss(pred, target):
    """pts to lines loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    eps = 1e-5
    loss = torch.abs(pred - target)
    # for polyline
    polyline_index = (target[:,0,:]!=target[:,-1,:]).any(dim=-1) # get polyline_index
    target_polyline = target[polyline_index, :, :]
    target_middle_s_line = torch.roll(target_polyline, 1, 1)
    A=target_middle_s_line[:, 1:-1, 1] - target_polyline[:, 1:-1, 1]
    B=target_polyline[:, 1:-1, 0] - target_middle_s_line[:, 1:-1, 0]
    C=target_middle_s_line[:, 1:-1, 0] * target_polyline[:, 1:-1, 1] - target_polyline[:, 1:-1, 0] * target_middle_s_line[:, 1:-1, 1]
    distance1=torch.abs(A*pred[polyline_index, 1:-1, 0]+B*pred[polyline_index, 1:-1, 1]+C)/(2 * torch.sqrt(A*A+B*B + eps))

    target_middle_n_line = torch.roll(target_polyline, -1, 1)
    A=target_middle_n_line[:, 1:-1, 1] - target_polyline[:, 1:-1, 1]
    B=target_polyline[:, 1:-1, 0] - target_middle_n_line[:, 1:-1, 0]
    C=target_middle_n_line[:, 1:-1, 0] * target_polyline[:, 1:-1, 1] - target_polyline[:, 1:-1, 0] * target_middle_n_line[:, 1:-1, 1]
    distance2=torch.abs(A*pred[polyline_index, 1:-1, 0]+B*pred[polyline_index, 1:-1, 1]+C)/(2 * torch.sqrt(A*A+B*B + eps))
    loss[polyline_index, 1:-1, :] = torch.stack((distance1, distance2), -1)

    # for polygon
    polygon_index = (target[:,0,:]==target[:,-1,:]).all(dim=-1) # get polygon_index
    # distance1
    target_polygon = target[polygon_index,:-1,:]
    target_middle_s_gon = torch.roll(target_polygon, 1, 1)
    A = target_middle_s_gon[:, :, 1] - target_polygon[:, :, 1]
    B = target_polygon[:, :, 0] - target_middle_s_gon[:, :, 0]
    C = target_middle_s_gon[:, :, 0] * target_polygon[:, :, 1] - target_polygon[:, :, 0] * target_middle_s_gon[:, :, 1]
    A, B, C = torch.cat((A, A[:, 0].unsqueeze(1)), dim=1), torch.cat((B, B[:, 0].unsqueeze(1)), dim=1), torch.cat((C, C[:, 0].unsqueeze(1)), dim=1)
    distance1=torch.abs(A*pred[polygon_index, :, 0]+B*pred[polygon_index, :, 1]+C)/(2 * torch.sqrt(A*A+B*B + eps))

    # distance2
    target_middle_n_gon = torch.roll(target_polygon, -1, 1)
    A = target_middle_n_gon[:, :, 1] - target_polygon[:, :, 1]
    B = target_polygon[:, :, 0] - target_middle_n_gon[:, :, 0]
    C = target_middle_n_gon[:, :, 0] * target_polygon[:, :, 1] - target_polygon[:, :, 0]* target_middle_n_gon[:, :, 1]
    A, B, C = torch.cat((A, A[:, 0].unsqueeze(1)), dim=1), torch.cat((B, B[:, 0].unsqueeze(1)), dim=1), torch.cat((C, C[:, 0].unsqueeze(1)), dim=1)
    distance2=torch.abs(A*pred[polygon_index, :, 0]+B*pred[polygon_index, :, 1]+C)/(2 * torch.sqrt(A*A+B*B + eps))
    
    loss[polygon_index, :, :] = torch.stack((distance1, distance2), -1)
    return loss

@LOSSES.register_module()
class Pts2LinesLoss(nn.Module):
    """point 2 line loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(Pts2LinesLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * pts_2_lines_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l2_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.pow(pred - target, 2)
    return loss

@LOSSES.register_module()
class PtsL2Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsL2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * pts_l2_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

def _get_max_dir_points_indices(dir_loss, dir_interval=1):
    """
    定位dir_loss最大值对应的关键帧点对索引
    Args:
        dir_loss (torch.Tensor): 方向损失，形状 [num_samples, num_dir]
            - num_samples：正样本数（需计算横向偏移的线实例数）
            - num_dir：每个样本的方向向量数（= num_pts - dir_interval）
        dir_interval (int): 生成方向向量的点间隔（如1=相邻点对，2=间隔1个点的点对）
            需与你计算dir_loss时的dir_interval保持一致（如之前代码中的self.dir_interval）
    Returns:
        start_indices (torch.Tensor): 关键帧点对的起始点索引，形状 [num_samples]
        end_indices (torch.Tensor): 关键帧点对的结束点索引，形状 [num_samples]
            （每个样本对应1组关键帧点对：(start_idx, end_idx)）
    """
    # 1. 找到每个样本中dir_loss最大的方向索引（num_dir维度的最大值位置）
    # max_dir_idx: [num_samples]，每个元素是该样本dir_loss最大的方向索引
    max_dir_values, max_dir_idx = torch.max(dir_loss, dim=1) # dim=1：在num_dir维度找最大值
    
    # 2. 根据方向索引和dir_interval，映射到对应的点对索引
    # 方向向量生成逻辑：dir_k = pts[k+dir_interval] - pts[k] → 对应点对 (k, k+dir_interval)
    start_indices = max_dir_idx  # 起始点索引 = 最大方向索引
    end_indices = max_dir_idx + dir_interval  # 结束点索引 = 最大方向索引 + 间隔
    
    return start_indices, end_indices, max_dir_values


def lateral_offset_loss(pred_pts, target_pts, pred_dir_pts,target_dir_pts, dir_interval,weight1=None,weight2=None,reduction='mean', avg_factor=None):
    """
    横向偏移Loss计算：基于dir_loss最大值定位关键帧点，计算点对点L1 Loss
    Args:
        pred_pts (torch.Tensor): 预测的点列，形状 [num_samples, num_pts, num_coords]
            - num_samples：正样本数
            - num_pts：每个样本的总点数
            - num_coords：坐标维度（2D=2，3D=3；横向偏移通常关注x或y维度，需确保坐标顺序正确）
        target_pts (torch.Tensor): 真实的点列，形状与pred_pts完全一致 [num_samples, num_pts, num_coords]
        dir_loss (torch.Tensor): 方向损失（PtsDirCosLoss的输出），形状 [num_samples, num_dir]
            - num_dir = num_pts - dir_interval（必须满足，否则点索引越界）
        dir_interval (int): 生成方向向量的点间隔（需与dir_loss计算时一致），默认1
        weight (torch.Tensor, optional): 点权重，形状 [num_samples, num_pts]
            - 用于过滤无效点（如padding点权重设为0），默认None（所有点有效）
        reduction (str): 损失归约方式，可选 'none'/'mean'/'sum'，默认 'mean'
        avg_factor (int, optional): 平均因子（用于mean归约），默认None（用有效点总数）
    
    Returns:
        loss (torch.Tensor): 横向偏移Loss（标量或逐样本Loss，取决于reduction）
    """
    # 1. 边界条件检查
    num_samples, num_pts, num_coords = pred_pts.shape
    dir_loss=pts_dir_cos_loss(pred_dir_pts,target_dir_pts,weight1, reduction='none', avg_factor=avg_factor)
    num_dir = dir_loss.shape[1]
    assert num_dir == num_pts - dir_interval, \
        f"dir_loss的num_dir={num_dir}需等于num_pts - dir_interval={num_pts - dir_interval}，否则点索引越界"
    assert pred_pts.shape == target_pts.shape, \
        f"pred_pts形状{pred_pts.shape}与target_pts形状{target_pts.shape}不匹配"
    
    # 2. 定位dir_loss最大值对应的关键帧点对索引
    start_indices, end_indices, max_dir_values = _get_max_dir_points_indices(dir_loss, dir_interval)
    
    # 3. 生成批量索引（用于提取每个样本的关键帧点）
    # batch_idx: [num_samples] → 每个样本的索引（如0,1,2,...,num_samples-1）
    batch_idx = torch.arange(num_samples, device=pred_pts.device)
    
    # 4. 提取关键帧点的预测和真实坐标
    # 提取起始点坐标：pred_start_pts.shape = [num_samples, num_coords]
    pred_start_pts = pred_pts[batch_idx, start_indices, :]
    target_start_pts = target_pts[batch_idx, start_indices, :]
    
    # 提取结束点坐标：pred_end_pts.shape = [num_samples, num_coords]
    pred_end_pts = pred_pts[batch_idx, end_indices, :]
    target_end_pts = target_pts[batch_idx, end_indices, :]
    
    # 5. 计算关键帧点对的点对点L1 Loss（可选择仅计算横向维度，如x维度或y维度）
    # 若为2D场景，横向偏移通常对应x维度（可根据你的坐标系调整，如改为[:,1:]计算y维度）
    # 这里计算所有坐标维度的L1（若需仅横向，可改为 pred_start_pts[:, 0:1] - target_start_pts[:, 0:1]）
    start_l1 = torch.abs(pred_start_pts - target_start_pts)  # [num_samples, num_coords]
    end_l1 = torch.abs(pred_end_pts - target_end_pts)        # [num_samples, num_coords]
    if weight2 is not None:
        # 提取关键帧点的权重：start_weight.shape = [num_samples]
        start_weight = weight2[batch_idx, start_indices]
        end_weight = weight2[batch_idx, end_indices]
    # 合并起始点和结束点的L1 Loss（每个样本的横向偏移Loss = 起始点L1 + 结束点L1）
    per_sample_loss = ((start_l1.sum(dim=1)*start_weight + end_l1.sum(dim=1)*end_weight))*max_dir_values # [num_samples]，sum(dim=1)合并坐标维度
    
    # 6. 应用点权重（过滤无效点）
    
        # 加权：无效点（权重0）的Loss会被过滤
    
    # 7. 损失归约
    loss  = per_sample_loss.sum()/avg_factor
    
    return loss


@LOSSES.register_module()
class LateralOffsetLoss(nn.Module):
    """
    横向偏移Loss的MMCV封装类（兼容框架训练流程）
    Args:
        dir_interval (int): 生成方向向量的点间隔，必须与dir_loss计算时一致，默认1
        reduction (str): 损失归约方式，可选 'none'/'mean'/'sum'，默认 'mean'
        loss_weight (float): 损失权重（平衡与其他Loss的重要性），默认1.0
    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(LateralOffsetLoss, self).__init__() 
        self.reduction = reduction        # 归约方式
        self.loss_weight = loss_weight    # 损失权重

    def forward(self,
                pred_pts,
                target_pts,
                pred_dir_pts,
                target_dir_pts, 
                dir_interval,
                weight1=None,
                weight2=None,
                avg_factor=None,
                reduction_override=None):
        """
        前向传播（符合MMCV Loss类接口规范）
        Args:
            pred_pts (torch.Tensor): 预测点列 [num_samples, num_pts, num_coords]
            target_pts (torch.Tensor): 真实点列 [num_samples, num_pts, num_coords]
            dir_loss (torch.Tensor): 方向损失 [num_samples, num_dir]
            weight (torch.Tensor, optional): 点权重 [num_samples, num_pts]，默认None
            avg_factor (int, optional): 平均因子，默认None
            reduction_override (str, optional): 覆盖默认归约方式，默认None
        
        Returns:
            loss (torch.Tensor): 加权后的横向偏移Loss（标量）
        """
        # 确定最终归约方式（优先级：reduction_override > self.reduction）
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        # 计算基础横向偏移Loss
        base_loss = lateral_offset_loss(
            pred_pts=pred_pts,
            target_pts=target_pts,
            pred_dir_pts=pred_dir_pts,
            target_dir_pts=target_dir_pts,
            dir_interval=dir_interval,
            weight1=weight1,
            weight2=weight2,
            reduction=reduction,
            avg_factor=avg_factor
        )
        
        # 应用损失权重（平衡与其他Loss的占比）
        loss = self.loss_weight * base_loss
        
        return loss
