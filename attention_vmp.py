import math
import torch
import torch.nn as nn
from attention import get_positional_encoding
from vmp import quadratic_vmp, relu_vmp, sigmoid_vmp, softmax_vmp, residual_vmp, LinearVMP, LayerNormVMP


class AttentionHeadVMP(nn.Module):
    """
    Multi Head Attention with skip connection
    h: nb of heads
    d: input dimension
    mode: residual connection mode for variance computation : identity, independence, taylor
    """
    def __init__(self, h, d, device, mode='identity', var_init=(1e-3, 1e-2)):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd = math.sqrt(d)
        self.h = h
        self.device = device
        self.mode = mode
        self.query = LinearVMP(d, h*(d//h), bias=False, var_init=var_init)
        self.key = LinearVMP(d, h*(d//h), bias=False, var_init=var_init)
        self.value = LinearVMP(d, h*(d//h), bias=False, var_init=var_init)

    def forward(self, x, var_x, masking=False):
        # b: batch size, l: sequence length
        q, var_q = self.query(x, var_x)  # b x l x h.s where s = d//h
        q = q.reshape(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        var_q = var_q.reshape(var_q.shape[0], var_q.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        k, var_k = self.key(x, var_x)  # b x l x h.s
        k = k.reshape(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        var_k = var_k.reshape(var_k.shape[0], var_k.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        v, var_v = self.value(x, var_x)  # b x l x h.s
        v = v.reshape(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        var_v = var_v.reshape(var_v.shape[0], var_v.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        a, var_a = quadratic_vmp(q, var_q, k.transpose(2, 3), var_k.transpose(2, 3))  # b x h x l x l
        a, var_a = a/self.rd, var_a/self.rd**2
        if masking:
            mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            a = a + mask
        a, var_a = softmax_vmp(a, var_a)  # b x h x l x l
        a, var_a = quadratic_vmp(a, var_a, v, var_v)
        a, var_a = a.transpose(1, 2), var_a.transpose(1, 2)  # b x l x h x s
        return residual_vmp(x, var_x, a.reshape(a.shape[0], a.shape[1], -1),
                            var_a.reshape(var_a.shape[0], var_a.shape[1], -1), mode=self.mode)  # b x l x h.s


class FinalHeadVMP(nn.Module):
    """
    Last Multi Head of the encoder
    """
    def __init__(self, h, d, var_init=(1e-3, 1e-2)):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.h = h
        self.key = LinearVMP(d, h*(d//h), bias=False, var_init=var_init)
        self.value = LinearVMP(d, h*(d//h), bias=False, var_init=var_init)

    def forward(self, x, var_x):
        k, var_k = self.key(x, var_x)
        k = k.reshape(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)
        var_k = var_k.reshape(var_k.shape[0], var_k.shape[1], self.h, -1).transpose(1, 2)
        v, var_v = self.value(x, var_x)
        v = v.reshape(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)
        var_v = var_v.reshape(var_v.shape[0], var_v.shape[1], self.h, -1).transpose(1, 2)
        return k, var_k, v, var_v


class DecoderHeadVMP(nn.Module):
    """
    Multi Head Attention using key and value from the encoder
    """
    def __init__(self, h, d, device, mode='identity', var_init=(1e-3, 1e-2)):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd = math.sqrt(d)
        self.h = h
        self.device = device
        self.mode = mode
        self.query = LinearVMP(d, h*(d//h), bias=False, var_init=var_init)

    def forward(self, x, var_x, k, var_k, v, var_v):
        q, var_q = self.query(x, var_x)
        q = q.reshape(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)
        var_q = var_q.reshape(var_q.shape[0], var_q.shape[1], self.h, -1).transpose(1, 2)
        a, var_a = quadratic_vmp(q, var_q, k.transpose(2, 3), var_k.transpose(2, 3))
        a, var_a = a/self.rd, var_a/self.rd**2
        mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        a = a + mask
        a, var_a = softmax_vmp(a, var_a)
        a, var_a = quadratic_vmp(a, var_a, v, var_v)
        a, var_a = a.transpose(1, 2), var_a.transpose(1, 2)
        return residual_vmp(x, var_x, a.reshape(a.shape[0], a.shape[1], -1),
                            var_a.reshape(var_a.shape[0], var_a.shape[1], -1), mode=self.mode)


class EncoderVMP(nn.Module):
    def __init__(self, param, device, encode=True):
        """
        n: nb of Multi Head Attention
        h: nb of heads per Multi Head
        k: embedding dimension
        """
        super().__init__()
        n, h = param['dim']
        self.n = n
        self.mode = param['residual']
        self.encode = encode
        self.embed = param['embed']
        # Use embedding layers or not
        if self.embed:
            emb_dim = param['emb']
            k = emb_dim[-1]
            self.nb_emb = len(emb_dim)
            self.emb = nn.ModuleList()
            self.emb.append(LinearVMP(param['input_dim'], emb_dim[0], var_init=param['var_init']))
            for i in range(len(emb_dim) - 1):
                self.emb.append(LinearVMP(emb_dim[i], emb_dim[i + 1], var_init=param['var_init']))
        else:
            k = param['input_dim']
        self.pos = get_positional_encoding(k, param['t_in'], device)
        self.multi = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(n-1):
            if i != 0:
                self.norm1.append(LayerNormVMP(k, var_init=param['var_init'], tol=param['tol']))
            self.multi.append(AttentionHeadVMP(h, k, device, self.mode, var_init=param['var_init']))
            self.norm2.append(LayerNormVMP(k, var_init=param['var_init'], tol=param['tol']))
            self.fc.append(LinearVMP(k, k, var_init=param['var_init']))
        self.norm1.append(LayerNormVMP(k, var_init=param['var_init'], tol=param['tol']))
        # If used as encoder: use FinalHead
        if self.encode:
            self.multi.append(FinalHeadVMP(h, k, param['var_init']))
        # Else, if used for classification or prediction: use AttentionHead
        else:
            self.multi.append(AttentionHeadVMP(h, k, device, self.mode, var_init=param['var_init']))

    def forward(self, x):
        var_x = torch.zeros_like(x)
        if self.embed:
            for i in range(self.nb_emb-1):
                x, var_x = relu_vmp(*self.emb[i](x, var_x))
            x, var_x = self.emb[-1](x, var_x)
        x = x + self.pos
        for i in range(self.n-1):
            # LayerNorms are before the layers
            if i != 0:
                x, var_x = self.norm1[i-1](x, var_x)
            x, var_x = self.multi[i](x, var_x)
            x, var_x = self.norm2[i](x, var_x)
            # Linear layer with skip connection
            x0, var_x0 = x[:], var_x[:]
            x, var_x = relu_vmp(*self.fc[i](x, var_x))
            x, var_x = residual_vmp(x0, var_x0, x, var_x, mode=self.mode)
        x, var_x = self.norm1[self.n-2](x, var_x)
        return self.multi[self.n-1](x, var_x)


class DecoderVMP(nn.Module):
    def __init__(self, param, device):
        """
        n: nb of Multi Head Attention
        h: nb of heads per Multi Head
        k: embedding dimension
        """
        super().__init__()
        n, h = param['dim']
        self.n = n
        self.device = device
        self.mode = param['residual']
        self.embed = param['embed']
        # Use embedding layers or not
        if self.embed:
            emb_dim = param['emb']
            k = emb_dim[-1]
            self.nb_emb = len(emb_dim)
            self.emb = nn.ModuleList()
            self.emb.append(LinearVMP(param['output_dim'], emb_dim[0], var_init=param['var_init']))
            for i in range(len(emb_dim) - 1):
                self.emb.append(LinearVMP(emb_dim[i], emb_dim[i+1], var_init=param['var_init']))
        else:
            k = param['output_dim']
        self.pos = get_positional_encoding(k, param['t_out']+1, device)
        self.multi1 = nn.ModuleList()
        self.multi2 = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.norm3 = nn.ModuleList()
        for i in range(n):
            if i != 0:
                self.norm1.append(LayerNormVMP(k, var_init=param['var_init'], tol=param['tol']))
            self.multi1.append(AttentionHeadVMP(h, k, device, self.mode, var_init=param['var_init']))
            self.norm2.append(LayerNormVMP(k, var_init=param['var_init'], tol=param['tol']))
            self.multi2.append(DecoderHeadVMP(h, k, device, self.mode, var_init=param['var_init']))
            self.norm3.append(LayerNormVMP(k, var_init=param['var_init'], tol=param['tol']))
            self.fc.append(LinearVMP(k, k, var_init=param['var_init']))
        if self.embed:
            self.fc.append(LinearVMP(k, param['output_dim'], var_init=param['var_init']))

    def forward(self, x, k, var_k, v, var_v):
        var_x = torch.zeros_like(x)
        if self.embed:
            for i in range(self.nb_emb-1):
                x, var_x = relu_vmp(*self.emb[i](x, var_x))
            x, var_x = self.emb[-1](x, var_x)
        x = x + self.pos
        for i in range(self.n):
            # LayerNorms are before the layers
            if i != 0:
                x, var_x = self.norm1[i-1](x, var_x)
            x, var_x = self.multi1[i](x, var_x, masking=True)
            x, var_x = self.norm2[i](x, var_x)
            x, var_x = self.multi2[i](x, var_x, k, var_k, v, var_v)
            x, var_x = self.norm3[i](x, var_x)
            # Linear layer with skip connection
            x0, var_x0 = x[:], var_x[:]
            x, var_x = relu_vmp(*self.fc[i](x, var_x), return_jac=True)
            x, var_x = residual_vmp(x0, var_x0, x, var_x, mode=self.mode)
            if self.embed:
                x, var_x = self.fc[-1](x, var_x)
        return sigmoid_vmp(x, var_x)


class TransformerVMP(nn.Module):
    """
    Inputs x and y must be normalized in [0,1]
    y must have dimension batch_size x (t_out - 1) x output_dim
    """
    def __init__(self, param, device):
        super().__init__()
        self.t_out = param['t_out']
        self.c = param['output_dim']
        self.device = device
        self.encoder = EncoderVMP(param, device)
        self.decoder = DecoderVMP(param, device)

    def forward(self, x, y):
        k, var_k, v, var_v = self.encoder(x)
        # The value -1 is a start token (since normal values are in [0,1])
        start_token = -torch.ones(x.shape[0], 1, self.c).to(self.device)
        # Append the start token to y
        y = torch.cat(tensors=(start_token, y), dim=1)
        return self.decoder(y, k, var_k, v, var_v)

    def inference(self, x):
        k, var_k, v, var_v = self.encoder(x)
        # Initialize prediction to -1 (start token)
        prediction = -torch.ones(x.shape[0], self.t_out, self.c).to(self.device)
        var_prediction = torch.ones(x.shape[0], self.t_out, self.c).to(self.device)
        for t in range(self.t_out):
            y, var_y = self.decoder(prediction, k, var_k, v, var_v)
            if t != self.t_out - 1:
                # The timestep index of prediction is offset by 1 because of the start token
                prediction[:, t+1, :] = y[:, t, :]
            else:
                # Remove the start token
                prediction = prediction[:, 1:, :]
                # Append the last predicted timestep at the end of prediction
                prediction = torch.cat(tensors=(prediction, y[:, -1, :].unsqueeze(1)), dim=1)
            var_prediction[:, t, :] = var_y[:, t, :]
        return prediction, var_prediction


class TransformerClassifierVMP(nn.Module):
    """
    Transformer Encoder for classification
    """
    def __init__(self, param, device):
        super().__init__()
        clas_dim = param['clas']
        self.nb_clas = len(clas_dim)
        self.encoder = EncoderVMP(param, device, encode=False)
        self.clas = nn.ModuleList()
        for i in range(self.nb_clas-1):
            self.clas.append(LinearVMP(clas_dim[i], clas_dim[i+1], var_init=param['var_init']))
        self.classifier = LinearVMP(clas_dim[-1], param['output_dim'], var_init=param['var_init'])

    def forward(self, x):
        x, var_x = self.encoder(x)
        # Global average pooling
        t_in = var_x.shape[1]
        x = x.mean(dim=1)
        var_x = var_x.mean(dim=1)/t_in
        for i in range(self.nb_clas-1):
            x, var_x = relu_vmp(*self.clas[i](x, var_x))
        return self.classifier(x, var_x)
