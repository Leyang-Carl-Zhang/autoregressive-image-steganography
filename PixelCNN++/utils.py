import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def concat_elu(x):
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


def log_sum_exp(x):
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def down_shift(x, pad=None):
    xs = [int(y) for y in x.size()]
    x = x[:, :, :xs[2] - 1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    xs = [int(y) for y in x.size()]
    x = x[:, :, :, :xs[3] - 1]
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def discretized_mix_logistic_loss(x, l):
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix], device=x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    return -torch.sum(log_sum_exp(log_probs))


def sample_from_discretized_mix_logistic(l, nr_mix):
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])

    temp = torch.empty(logit_probs.size(), device=l.device, dtype=l.dtype).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    u = torch.empty(means.size(), device=l.device, dtype=l.dtype).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x0 = torch.clamp(x[:, :, :, 0], -1.0, 1.0)
    x1 = torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.0, 1.0)
    x2 = torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.0, 1.0)
    out = torch.cat([x0.unsqueeze(3), x1.unsqueeze(3), x2.unsqueeze(3)], dim=3)
    return out.permute(0, 3, 1, 2)


# discretized-logistic bin masses for all 256 levels
def _logistic_bin_probs_1d(values, mean, log_scale):
    centered = values[:, None] - mean[None, :]
    inv_stdv = torch.exp(-log_scale)[None, :]
    plus_in = inv_stdv * (centered + 1.0 / 255.0)
    min_in = inv_stdv * (centered - 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered
    log_pdf_mid = mid_in - log_scale[None, :] - 2.0 * F.softplus(mid_in)
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (values[:, None] > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (values[:, None] < -0.999).float()
    return cond * log_cdf_plus + (1.0 - cond) * inner_out


# 256-way discrete pmf for one rgb channel
def discretized_mix_logistic_pmf_channel(logits, channel, x_r=None, x_g=None):
    nr_mix = logits.shape[0] // 10
    logit_probs = logits[:nr_mix]
    params = logits[nr_mix:].view(3, 3 * nr_mix)
    means = params[:, :nr_mix]
    log_scales = torch.clamp(params[:, nr_mix:2 * nr_mix], min=-7.)
    coeffs = torch.tanh(params[:, 2 * nr_mix:3 * nr_mix])
    weights = torch.softmax(logit_probs, dim=0)
    values = torch.linspace(-1.0, 1.0, 256, device=logits.device, dtype=logits.dtype)

    if channel == 0:
        mu = means[0]
    elif channel == 1:
        if x_r is None:
            raise ValueError('x_r is required for channel 1.')
        mu = means[1] + coeffs[0] * x_r
    elif channel == 2:
        if x_r is None or x_g is None:
            raise ValueError('x_r and x_g are required for channel 2.')
        mu = means[2] + coeffs[1] * x_r + coeffs[2] * x_g
    else:
        raise ValueError('channel must be 0, 1, or 2')

    per_mix = _logistic_bin_probs_1d(values, mu, log_scales[channel])
    probs = (per_mix * weights[None, :]).sum(dim=1)
    probs = torch.clamp(probs, min=1e-12)
    return probs / probs.sum()