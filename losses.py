from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as tf
import scipy.misc
import os
import numpy as np

from flowbias.data_manipulation.extract_minor_flow import compute_secondary_flows


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_avg_pool2d(inputs, [h, w])

def _downsample_mask_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_max_pool2d(inputs, [h, w])

def _upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)

def fbeta_score(y_true, y_pred, beta, eps=1e-8):
    beta2 = beta ** 2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=2).sum(dim=2)
    precision = true_positive / (y_pred.sum(dim=2).sum(dim=2) + eps)
    recall = true_positive / (y_true.sum(dim=2).sum(dim=2) + eps)

    return torch.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))

def f1_score_bal_loss(y_pred, y_true):
    eps = 1e-8

    tp = -(y_true * torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    fn = -((1 - y_true) * torch.log((1 - y_pred) + eps)).sum(dim=2).sum(dim=2).sum(dim=1)

    denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps
    denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps

    return ((tp / denom_tp).sum() + (fn / denom_fn).sum()) * y_pred.size(2) * y_pred.size(3) * 0.5


class MultiScaleEPE_FlowNet(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet, self).__init__()
        self._args = args        
        self._batch_size = args.batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        b, _, _, _ = target_dict["target1"].size()
        bs_full = self._batch_size if self._batch_size is not None else b

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                target_i = _downsample2d_as(target, output_i)
                epe_i = _elementwise_epe(output_i, target_i)
                total_loss = total_loss + self._weights[i] * epe_i.sum() / bs_full
                loss_dict["epe%i" % (i + 2)] = epe_i.mean()
            loss_dict["total_loss"] = total_loss
        else:
            output = output_dict["flow1"]
            target = target_dict["target1"]
            epe = _elementwise_epe(output, target)
            # FIXME multipy by (local_batch_size / batch_size) to account for virtual batch size
            loss_dict["epe"] = epe.mean()

        return loss_dict


class MultiScaleSparseEPE_FlowNet(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleSparseEPE_FlowNet, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]

    def forward(self, output_dict, target_dict):
        raise RuntimeError("implement correct masking")

        loss_dict = {}

        valid_masks = target_dict["input_valid"]
        b, _, h, w = target_dict["target1"].size()
        bs_full = self._batch_size if self._batch_size is not None else b

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                valid_mask = _downsample_mask_as(valid_masks, output_ii)
                masked_epe = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii))[valid_mask != 0]
                norm_const = (h * w) / (valid_mask[ii, ...].sum())

                total_loss += self._weights[ii] * (masked_epe.sum() * norm_const) / bs_full
                loss_dict["epe%i" % (ii + 2)] = (masked_epe.mean() * norm_const)
            loss_dict["total_loss"] = total_loss
        else:
            output = output_dict["flow1"]
            target = target_dict["target1"]
            epe = _elementwise_epe(output, target)
            loss_dict["epe"] = epe.mean() * (b / bs_full)

        return loss_dict

class MultiScaleEPE_PWC(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        b, _, _, _ = target_dict["target1"].size()
        bs_full = self._batch_size if self._batch_size is not None else b

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] = total_loss / bs_full
        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            # FIXME multipy by (local_batch_size / batch_size) to account for virtual batch size
            loss_dict["epe"] = epe.mean()

        return loss_dict


class MultiScaleSparseEPE_PWC(nn.Module):
    def __init__(self, args):
        super(MultiScaleSparseEPE_PWC, self).__init__()
        self._args = args
        self._batch_size = args.batch_size

        # corrected weights, as the image is now upsampled
        # -> we have more (upsampled pixels, so each pixel is worth less
        self._up_weights = [0.32/(4**6), 0.08/(4**5), 0.02/(4**4), 0.01/(4**3), 0.005/(4**2)]
        #self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        valid_masks = target_dict["input_valid"]
        b, _, h, w = target_dict["target1"].size()
        bs_full = self._batch_size if self._batch_size is not None else b

        if self.training:
            output_flo = output_dict['flow']

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(output_flo):
                # compute dense epe
                output_ii_up = _upsample2d_as(output_ii, valid_masks)
                dense_epe = _elementwise_epe(output_ii_up, target)

                # if there are less valid pixels, each pixel is 'worth' more
                norm_const = (h * w) / (valid_masks.view(b, -1).sum(1))

                # select the valid pixels for each image and correct their weight
                for bb in range(b):
                    masked_epe_bb = dense_epe[bb, :, :, :][valid_masks[bb, :, :, :] != 0]
                    total_loss += self._up_weights[ii] * masked_epe_bb.sum() * norm_const[bb]
            loss_dict["total_loss"] = total_loss / bs_full
        else:
            flow_epe = _elementwise_epe(output_dict["flow"], target_dict["target1"]) * valid_masks
            epe_per_image = (flow_epe.view(b, -1).sum(1)) / (valid_masks.view(b, -1).sum(1))
            loss_dict["epe"] = epe_per_image.mean() * (b/bs_full)

        return loss_dict


class MultiScaleAdaptiveEPE_PWC(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.denseLoss = MultiScaleEPE_PWC(args)
        self.sparseLoss = MultiScaleSparseEPE_PWC(args)

    def forward(self, output_dict, target_dict):
        if target_dict["dense"][0]:
            return self.denseLoss.forward(output_dict, target_dict)
        else:
            return self.sparseLoss.forward(output_dict, target_dict)


class MultiScaleEPE_SecondaryFlow_PWC(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_SecondaryFlow_PWC, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

        self._em_iterations = 5
        self._sec_flow_weight = 0.6


    def forward(self, output_dict, target_dict):
        loss_dict = {}

        b, _, _, _ = target_dict["target1"].size()
        bs_full = self._batch_size if self._batch_size is not None else b

        if self.training:
            outputs = output_dict['flow']
            outputs_sec = output_dict['sec_flow']

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]
            with torch.no_grad():
                target_primary, target_secondary = compute_secondary_flows(target, self._em_iterations)

            #primary flow
            total_loss = 0
            for ii, (output_ii, target_ii) in enumerate(zip(outputs, target_primary)):
                #loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                loss_ii = _elementwise_epe(output_ii, target_ii).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] = total_loss / bs_full

            # secondary flow
            total_loss = 0
            for ii, (output_ii, target_ii) in enumerate(zip(outputs_sec, target_secondary)):
                #loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                loss_ii = _elementwise_epe(output_ii, target_ii).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] += (total_loss / bs_full) * self._sec_flow_weight

        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            # FIXME multipy by (local_batch_size / batch_size) to account for virtual batch size
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleAdaptiveEPE_SecondaryFlow_PWC(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.denseLoss = MultiScaleEPE_SecondaryFlow_PWC(args)
        #FIXME implement sparse secondary flow ...
        self.sparseLoss = MultiScaleSparseEPE_PWC(args)

    def forward(self, output_dict, target_dict):
        if target_dict["dense"][0]:
            return self.denseLoss.forward(output_dict, target_dict)
        else:
            return self.sparseLoss.forward(output_dict, target_dict)

class MultiScaleEPE_PWCDelta(nn.Module):
    def __init__(self,
                 args):
        super(MultiScaleEPE_PWCDelta, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def flow_gradient(self, x):
        b, _, w, h = x.size()
        out = torch.empty((b, 2, w, h))

        out[:, 0, :-1, :] = x[:, 0, 1:, :] - x[:, 0, :-1, :]
        out[:, 0, :, :-1] = x[:, 1, :, 1:] - x[:, 1, :, :-1]
        x[:, 0, :, -1] = 0
        x[:, 1, -1, :] = 0
        return x

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        target_delta = self.flow_gradient(target_dict["target1"])[:, :, :-1, :-1]

        b, _, _, _ = target_dict["target1"].size()
        bs_full = self._batch_size if self._batch_size is not None else b

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target = self._args.model_div_flow * target_delta

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] = total_loss / bs_full

        else:
            epe = _elementwise_epe(output_dict["flow"][:, :, :-1, :-1], target_delta)
            loss_dict["epe"] = epe.mean()
            # FIXME multipy by (local_batch_size / batch_size) to account for virtual batch size

        return loss_dict


class L1ConnectorLoss(nn.Module):
    def __init__(self, args):

        super(L1ConnectorLoss, self).__init__()
        self._args = args
        self._batch_size = args.batch_size

        self.l1loss = nn.L1Loss()

    def forward(self, output_dict, target_dict):
        target_x1_0 = self.l1loss(output_dict['x1_0'], target_dict["target_x1_0"])
        target_x1_1 = self.l1loss(output_dict['x1_1'], target_dict["target_x1_1"])
        target_x1_2 = self.l1loss(output_dict['x1_2'], target_dict["target_x1_2"])
        target_x1_3 = self.l1loss(output_dict['x1_3'], target_dict["target_x1_3"])
        target_x1_4 = self.l1loss(output_dict['x1_4'], target_dict["target_x1_4"])

        loss_dict = {
            "x1_0": target_x1_0 / self._batch_size,
            "x1_1": target_x1_1 / self._batch_size,
            "x1_2": target_x1_2 / self._batch_size,
            "x1_3": target_x1_3 / self._batch_size,
            "x1_4": target_x1_4 / self._batch_size,
            "total_loss": (target_x1_0 + target_x1_1 + target_x1_2 + target_x1_3 + target_x1_4) / 5
        }
        return loss_dict


class MSEConnectorLoss(nn.Module):
    def __init__(self, args):

        super(MSEConnectorLoss, self).__init__()
        self._args = args
        self._batch_size = args.batch_size

        self.MSEloss = nn.MSELoss()

    def forward(self, output_dict, target_dict):
        target_x1_0 = self.MSEloss(output_dict['x1_0'], target_dict["target_x1_0"])
        target_x1_1 = self.MSEloss(output_dict['x1_1'], target_dict["target_x1_1"])
        target_x1_2 = self.MSEloss(output_dict['x1_2'], target_dict["target_x1_2"])
        target_x1_3 = self.MSEloss(output_dict['x1_3'], target_dict["target_x1_3"])
        target_x1_4 = self.MSEloss(output_dict['x1_4'], target_dict["target_x1_4"])

        loss_dict = {
            "x1_0": target_x1_0 / self._batch_size,
            "x1_1": target_x1_1 / self._batch_size,
            "x1_2": target_x1_2 / self._batch_size,
            "x1_3": target_x1_3 / self._batch_size,
            "x1_4": target_x1_4 / self._batch_size,
            "total_loss": (target_x1_0 + target_x1_1 + target_x1_2 + target_x1_3 + target_x1_4) / 5
        }
        return loss_dict

