import math
import torch
import torch.nn as nn


def get_positional_encoding(d, t_in, device):
    if d % 2 == 0:
        d2 = d//2
    else:
        d2 = d//2 + 1
    pos = torch.zeros(t_in, 2*d2).to(device)
    for t in range(t_in):
        pos[t, :] = t + 1
    omega = torch.ones(1, d2).to(device)
    for k in range(d2):
        omega[0, k] = 1/1000**(k/d2)
    omega = omega.repeat_interleave(2, dim=1)
    pos = pos*omega
    phase = torch.tensor([0, torch.pi/2]).to(device)
    phase = phase.repeat(d2).unsqueeze(0)
    pos = torch.sin(pos + phase)
    if d % 2 == 0:
        return pos
    else:
        return pos[:, :-1]


class AttentionHead(nn.Module):
    """
    Multi Head Attention with skip connection
    h: nb of heads
    d: input dimension
    """
    def __init__(self, h, d, device):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd = math.sqrt(d)
        self.h = h
        self.device = device
        self.query = nn.Linear(d, h*(d//h), bias=False)
        self.key = nn.Linear(d, h*(d//h), bias=False)
        self.value = nn.Linear(d, h*(d//h), bias=False)

    def forward(self, x, masking=False):
        q = self.query(x)  # b x l x h.s where s = d//h
        q = q.reshape(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        k = self.key(x)  # b x l x h.s
        k = k.reshape(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        v = self.value(x)  # b x l x h.s
        v = v.reshape(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)  # b x h x l x s
        a = torch.matmul(q, k.transpose(2, 3))/self.rd  # b x h x l x l
        if masking:
            mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            a = a + mask
        a = nn.functional.softmax(a, dim=3)  # b x h x l x l
        a = torch.matmul(a, v).transpose(dim0=1, dim1=2)  # b x l x h x s
        return x + a.reshape(a.shape[0], a.shape[1], -1)  # b x l x h.s


class FinalHead(nn.Module):
    """
    Last Multi Head of the encoder
    """
    def __init__(self, h, d):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.h = h
        self.key = nn.Linear(d, h*(d//h), bias=False)
        self.value = nn.Linear(d, h*(d//h), bias=False)

    def forward(self, x):
        k = self.key(x)
        k = k.reshape(k.shape[0], k.shape[1], self.h, -1).transpose(1, 2)
        v = self.value(x)
        v = v.reshape(v.shape[0], v.shape[1], self.h, -1).transpose(1, 2)
        return k, v


class DecoderHead(nn.Module):
    """
    Multi Head Attention using key and value from the encoder
    """
    def __init__(self, h, d, device):
        super().__init__()
        if d % h != 0:
            raise RuntimeError
        self.rd = math.sqrt(d)
        self.h = h
        self.device = device
        self.query = nn.Linear(d, h*(d//h), bias=False)

    def forward(self, x, k, v):
        q = self.query(x)
        q = q.reshape(q.shape[0], q.shape[1], self.h, -1).transpose(1, 2)
        a = torch.matmul(q, k.transpose(2, 3))/self.rd
        mask = torch.ones(*a.shape).triu(diagonal=1).to(self.device)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        a = a + mask
        a = nn.functional.softmax(a, dim=3)
        a = torch.matmul(a, v).transpose(dim0=1, dim1=2)
        return x + a.reshape(a.shape[0], a.shape[1], -1)


class Encoder(nn.Module):
    def __init__(self, param, device, encode=True):
        """
        n: nb of Multi Head Attention
        h: nb of heads per Multi Head
        d: input dimension
        k: embedding dimension
        """
        super().__init__()
        n, h = param['dim']
        self.n = n
        self.embed = param['embed']
        # Use embedding layers or not
        if self.embed:
            emb_dim = param['emb']
            k = emb_dim[-1]
            self.nb_emb = len(emb_dim)
            self.emb = nn.ModuleList()
            self.emb.append(nn.Linear(param['input_dim'], emb_dim[0]))
            for i in range(self.nb_emb-1):
                self.emb.append(nn.Linear(emb_dim[i], emb_dim[i+1]))
        else:
            k = param['input_dim']
        self.pos = get_positional_encoding(k, param['t_in'], device)
        self.multi = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(n-1):
            if i != 0:
                self.norm1.append(nn.LayerNorm(k))
            self.multi.append(AttentionHead(h, k, device))
            self.norm2.append(nn.LayerNorm(k))
            self.fc.append(nn.Linear(k, k))
        self.norm1.append(nn.LayerNorm(k))
        # If used as encoder: use FinalHead
        if encode:
            self.multi.append(FinalHead(h, k))
        # Else, if used for classification or prediction: use AttentionHead
        else:
            self.multi.append(AttentionHead(h, k, device))

    def forward(self, x):
        if self.embed:
            for i in range(self.nb_emb-1):
                x = nn.functional.relu(self.emb[i](x))
            x = self.emb[-1](x)
        x = x + self.pos
        for i in range(self.n-1):
            # LayerNorms are before the layers
            if i != 0:
                x = self.norm1[i-1](x)
            x = self.multi[i](x)
            x = self.norm2[i](x)
            # Linear layer with skip connection
            x = x + nn.functional.relu(self.fc[i](x))
        x = self.norm1[self.n-2](x)
        x = self.multi[self.n-1](x)
        return x


class Decoder(nn.Module):
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
        self.embed = param['embed']
        # Use embedding layers or not
        if self.embed:
            emb_dim = param['emb']
            k = emb_dim[-1]
            self.nb_emb = len(emb_dim)
            self.emb = nn.ModuleList()
            self.emb.append(nn.Linear(param['output_dim'], emb_dim[0]))
            for i in range(len(emb_dim) - 1):
                self.emb.append(nn.Linear(emb_dim[i], emb_dim[i + 1]))
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
                self.norm1.append(nn.LayerNorm(k))
            self.multi1.append(AttentionHead(h, k, device))
            self.norm2.append(nn.LayerNorm(k))
            self.multi2.append(DecoderHead(h, k, device))
            self.norm3.append(nn.LayerNorm(k))
            self.fc.append(nn.Linear(k, k))
        if self.embed:
            self.fc.append(nn.Linear(k, param['output_dim']))

    def forward(self, x, k, v):
        if self.embed:
            for i in range(self.nb_emb-1):
                x = nn.functional.relu(self.emb[i](x))
            x = self.emb[-1](x)
        x = x + self.pos
        for i in range(self.n):
            # LayerNorms are before the layers
            if i != 0:
                x = self.norm1[i-1](x)
            x = self.multi1[i](x, masking=True)
            x = self.norm2[i](x)
            x = self.multi2[i](x, k, v)
            x = self.norm3[i](x)
            # Linear layer with skip connection
            x = nn.functional.relu(self.fc[i](x)) + x
            if self.embed:
                x = self.fc[-1](x)
        return torch.sigmoid(x)


class TransformerED(nn.Module):
    """
        Inputs x and y must be normalized in [0,1]
        y must have dimension batch_size x (t_out - 1) x output_dim
    """
    def __init__(self, param, device):
        super().__init__()
        self.t_out = param['t_out']
        self.c = param['output_dim']
        self.device = device
        self.encoder = Encoder(param, device)
        self.decoder = Decoder(param, device)

    def forward(self, x, y):
        k, v = self.encoder(x)
        # The value -1 is a start token (since normal values are in [0,1])
        start_token = -torch.ones(x.shape[0], 1, self.c).to(self.device)
        # Append the start token to y
        y = torch.cat(tensors=(start_token, y), dim=1)
        return self.decoder(y, k, v)

    def inference(self, x):
        k, v = self.encoder(x)
        prediction = -torch.ones(x.shape[0], self.t_out, self.d).to(self.device)
        for t in range(self.t_out):
            y = self.decoder(prediction, k, v)
            if t != self.t_out - 1:
                # The timestep index of prediction is offset by 1 because of the start token
                prediction[:, t+1, :] = y[:, t, :]
            else:
                # Remove the start token
                prediction = prediction[:, 1:, :]
                # Append the last predicted timestep at the end of prediction
                prediction = torch.cat(tensors=(prediction, y[:, -1, :].unsqueeze(1)), dim=1)
        return prediction


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder for classification
    """
    def __init__(self, param, device):
        super().__init__()
        clas_dim = param['clas']
        self.nb_clas = len(clas_dim)
        self.encoder = Encoder(param, device, encode=False)
        self.clas = nn.ModuleList()
        for i in range(self.nb_clas-1):
            self.clas.append(nn.Linear(clas_dim[i], clas_dim[i+1]))
        self.classifier = nn.Linear(clas_dim[-1], param['output_dim'])

    def forward(self, x):
        x = self.encoder(x)
        # Global average pooling
        x = x.mean(dim=1)
        for i in range(self.nb_clas-1):
            x = nn.functional.relu(self.clas[i](x))
        return self.classifier(x)
