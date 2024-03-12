import math
import torch
import torch.nn as nn


def loss_vmp(logit, var_logit, target, model, param):
    # Compute the regularization term
    kl = 0
    sig_sum = 0
    sig_tot = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'rho' in name:
                sig = nn.functional.softplus(p) + param['tol']
                sig_sum += sig.sum().item()
                sig_tot += sig.numel()
                kl += (sig - torch.log(sig)).sum()
            else:
                kl += (p**2).sum()

    # Compute the expected negative log-likelihood
    prob, var_prob = softmax_vmp(logit, var_logit)
    target = nn.functional.one_hot(target, num_classes=param['output_dim'])
    var_prob += param['tol']
    inv_var = torch.div(input=1, other=var_prob)
    mse = (target - prob)**2
    batch_size = prob.shape[0]
    log_var = torch.log(var_prob).sum()/2/batch_size
    mse_inv = (mse*inv_var).sum()/2/batch_size

    return log_var + mse_inv, kl, var_prob, (sig_sum/sig_tot, log_var.item(), mse_inv.item())


def quadratic_vmp(x, var_x, y, var_y):
    return torch.matmul(x, y), torch.matmul(var_x + x**2, var_y) + torch.matmul(var_x, y**2)


def quadratic_jac(x, jac_x, y, jac_y):
    return torch.matmul(jac_x, y) + torch.matmul(x, jac_y)


def relu_vmp(x, var_x, return_jac=False):
    x = nn.functional.relu(x)
    der = torch.logical_not(torch.eq(x, other=0)).long()
    der = der.detach()
    if return_jac:
        return x, var_x*der, der
    else:
        return x, var_x*der


def sigmoid_vmp(x, var_x):
    x = torch.sigmoid(x)
    der = x*(1 - x)
    der = der.detach()
    return x, var_x*der**2


def softmax_vmp(x, var_x, return_jac=False):
    """
    To avoid an out-of-memory error, we must neglect the off-diagonal terms of the Jacobian
    """
    prob = nn.functional.softmax(x, dim=-1)
    der = prob*(1 - prob)
    der = der.detach()
    if return_jac:
        return prob, var_x*der**2, der
    else:
        return prob, var_x*der**2


def residual_vmp(x, var_x, f, var_f=None, jac=None, mode='independence'):
    if mode == 'taylor':
        if (var_f is None) or (jac is None):
            raise RuntimeError
        return x + f, torch.maximum(var_x + var_f + 2*(jac*(var_x + x**2) - x*f), torch.tensor(0))
    elif mode == 'independence':
        if var_f is None:
            raise RuntimeError
        return x + f, var_x + var_f
    elif mode == 'identity':
        return x + f, var_x
    else:
        raise NotImplementedError


class LinearVMP(nn.Module):
    def __init__(self, in_features, out_features, bias=True, var_init=(1e-3, 1e-2)):
        super().__init__()
        self.size_in, self.size_out, self.biased = in_features, out_features, bias
        weight = torch.zeros(out_features, in_features)
        rho = torch.zeros(out_features, in_features)
        self.weight = nn.Parameter(weight)
        self.rho = nn.Parameter(rho)
        bound = math.sqrt(6/(in_features + out_features))
        # Inversion of softplus to get the bounds of rho
        rho1, rho2 = math.log(math.exp(var_init[0]) - 1), math.log(math.exp(var_init[1]) - 1)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.rho, rho1, rho2)
        if bias:
            b = torch.zeros(out_features)
            b_rho = torch.zeros(out_features)
            self.bias = nn.Parameter(b)
            self.b_rho = nn.Parameter(b_rho)
            bound = 1/math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.uniform_(self.b_rho, rho1, rho2)

    def forward(self, mu, sigma):
        w_sig = nn.functional.softplus(self.rho)
        mu = mu.transpose(-1, -2)
        sigma = sigma.transpose(-1, -2)
        mean = torch.matmul(self.weight, mu)
        var = torch.matmul(w_sig + self.weight**2, sigma) + torch.matmul(w_sig, mu**2)
        if self.biased:
            return mean.transpose(-2, -1) + self.bias, var.transpose(-2, -1) + nn.functional.softplus(self.b_rho)
        else:
            return mean.transpose(-2, -1), var.transpose(-2, -1)

    def get_jac(self):
        return self.weight.transpose(-2, -1)


class LayerNormVMP(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True, var_init=(1e-3, 1e-2), tol=1e-9):
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
            rho1, rho2 = math.log(math.exp(var_init[0]) - 1), math.log(math.exp(var_init[1]) - 1)
            nn.init.uniform_(self.rho, rho1, rho2)
            nn.init.uniform_(self.b_rho, rho1, rho2)

    def forward(self, mu, sigma):
        mean = mu.mean(dim=-1, keepdim=True)
        var = mu.var(dim=-1, keepdim=True) + self.tol
        if self.elementwise_affine:
            return ((mu - mean)/torch.sqrt(var)*self.weight + self.bias,
                    sigma/var*nn.functional.softplus(self.rho) + nn.functional.softplus(self.b_rho))
        else:
            return (mu - mean)/torch.sqrt(var), sigma/var
