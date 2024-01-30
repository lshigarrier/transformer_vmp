import math
import torch
import torch.nn as nn


def clamp_nan(x, tol):
    return torch.clamp(torch.nan_to_num(x, nan=tol, posinf=1/tol, neginf=-1/tol), min=tol, max=1/tol)


def no_nan(x, tol):
    return torch.nan_to_num(x, nan=tol, posinf=1/tol, neginf=-1/tol)


def loss_vmp(prob, var_prob, target, model, param):
    # Compute the regularization term
    kl = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'rho' in name:
                sig = nn.functional.softplus(p)
                kl += no_nan(sig - torch.log(sig + param['tol']), param['tol']).sum()
            else:
                kl += no_nan(p**2, param['tol']).sum()

    # Compute the expected negative log-likelihood
    prob = no_nan(prob, param['tol']).clamp(min=param['tol'], max=1-param['tol'])
    target = nn.functional.one_hot(target, num_classes=param['output_dim'])
    var_prob = clamp_nan(var_prob, param['tol'])
    inv_var = torch.div(input=1, other=var_prob + param['tol'])
    nll = no_nan((torch.log(var_prob + param['tol']) + (target - prob)**2*inv_var), param['tol']).sum(dim=-1)
    nll = nll.sum()/prob.shape[0]

    return nll, kl


def quadratic_vmp(x, var_x, y, var_y, tol=1e-3):
    return torch.matmul(x, y), clamp_nan(torch.matmul(var_x + x**2, var_y) + torch.matmul(var_x, y**2), tol)


def quadratic_jac(x, jac_x, y, jac_y):
    return torch.matmul(jac_x, y) + torch.matmul(x, jac_y)


def relu_vmp(x, var_x, return_jac=False, tol=1e-3):
    x = nn.functional.relu(x)
    der = torch.logical_not(torch.eq(x, other=0)).long()
    if return_jac:
        return x, clamp_nan(var_x*der, tol), der
    else:
        return x, clamp_nan(var_x*der, tol)


def sigmoid_vmp(x, var_x, tol=1e-3):
    x = torch.sigmoid(x)
    der = x*(1 - x)
    return x, clamp_nan(var_x*der**2, tol)


def softmax_vmp(x, var_x, return_jac=False, tol=1e-3):
    """
    To avoid an out-of-memory error, we must neglect the off-diagonal terms of the Jacobian
    """
    prob = nn.functional.softmax(x, dim=-1)
    der = prob*(1 - prob)
    if return_jac:
        return prob, clamp_nan(var_x*der**2, tol), der
    else:
        return prob, clamp_nan(var_x*der**2, tol)


def residual_vmp(x, var_x, f, var_f=None, jac=None, mode='independence', tol=1e-3):
    if mode == 'taylor':
        if (var_f is None) or (jac is None):
            raise RuntimeError
        return x + f, clamp_nan(torch.maximum(var_x + var_f + 2*(jac*(var_x + x**2) - x*f), torch.tensor(0)), tol)
    elif mode == 'independence':
        if var_f is None:
            raise RuntimeError
        return x + f, clamp_nan(var_x + var_f, tol)
    elif mode == 'identity':
        return x + f, clamp_nan(var_x, tol)
    else:
        raise NotImplementedError


class LinearVMP(nn.Module):
    def __init__(self, in_features, out_features, bias=True, var_init=1e-8, tol=1e-3):
        super().__init__()
        self.size_in, self.size_out, self.biased, self.tol = in_features, out_features, bias, tol
        weight = torch.zeros(out_features, in_features)
        rho = torch.zeros(out_features, in_features)
        self.weight = nn.Parameter(weight)
        self.rho = nn.Parameter(rho)
        bound = math.sqrt(6/(in_features + out_features))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.rho, -var_init*bound, var_init*bound)
        if bias:
            b = torch.zeros(out_features)
            b_rho = torch.zeros(out_features)
            self.bias = nn.Parameter(b)
            self.b_rho = nn.Parameter(b_rho)
            bound = 1/math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.uniform_(self.b_rho, -var_init*bound, var_init*bound)

    def forward(self, mu, sigma):
        w_sig = nn.functional.softplus(self.rho)
        mu = mu.transpose(-1, -2)
        sigma = sigma.transpose(-1, -2)
        mean = torch.matmul(self.weight, mu)
        var = torch.matmul(w_sig + self.weight**2, sigma) + torch.matmul(w_sig, mu**2)
        if self.biased:
            return (mean.transpose(-2, -1) + self.bias,
                    clamp_nan(var.transpose(-2, -1) + nn.functional.softplus(self.b_rho), self.tol))
        else:
            return mean.transpose(-2, -1), clamp_nan(var.transpose(-2, -1), self.tol)

    def get_jac(self):
        return self.weight.transpose(-2, -1)


class LayerNormVMP(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True, var_init=1e-8, tol=1e-3):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.tol = tol
        if elementwise_affine:
            weight = torch.ones(normalized_shape)
            rho = torch.zeros(normalized_shape)
            b = torch.zeros(normalized_shape)
            b_rho = torch.zeros(normalized_shape)
            self.weight = nn.Parameter(weight)
            self.rho = nn.Parameter(rho)
            self.bias = nn.Parameter(b)
            self.b_rho = nn.Parameter(b_rho)
            bound = var_init / math.sqrt(normalized_shape)
            nn.init.uniform_(self.rho, -bound, bound)
            nn.init.uniform_(self.b_rho, -bound, bound)

    def forward(self, mu, sigma):
        mean = mu.mean(dim=-1, keepdim=True)
        var = mu.var(dim=-1, keepdim=True)
        if self.elementwise_affine:
            return ((mu - mean)/torch.sqrt(var + self.tol)*self.weight + self.bias,
                    clamp_nan(sigma/(var + self.tol)*nn.functional.softplus(self.rho) +
                              nn.functional.softplus(self.b_rho), self.tol))
        else:
            return ((mu - mean)/torch.sqrt(var + self.tol),
                    clamp_nan(sigma/(var + self.tol), self.tol))
