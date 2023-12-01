#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.item()
    # s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model, layers):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    layer_grads = {}
    layer_params = {}
    for layer in layers:
        layer_grads[layer] = []
        layer_params[layer] = []

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        params.append(param)
        grads.append(param.grad + 0.)
        for layer in layers:
            if layer in name:
                layer_params[layer].append(param)
                layer_grads[layer].append(param.grad + 0.)

    return params, grads, layer_grads, layer_params


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

param_layers = ['Linear', 'Bilinear', 'Conv1d', 'Conv2d', 'Conv3d', 'Conv2DTranspose', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']

def get_param_layers(model):
    found_layers = []
    for layer_name, module in model.named_modules():
        if module.__class__.__name__ in param_layers:
            found_layers.append(layer_name)
    return found_layers
