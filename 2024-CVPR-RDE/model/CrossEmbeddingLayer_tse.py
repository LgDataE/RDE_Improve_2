import torch
import torch.nn as nn
import torch.nn.functional as F
 
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results

def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)

def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x
 
class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio
        self.reliability_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1)
        )
        self.reliability_blend = 0.1

    def forward(self, features, text, atten):
        # print(atten) 64 x 77 x 77
        mask =  ((text != 0) + 0)
        lengths = mask.sum(1).view(-1) - 2 # -2 for SOS token and EOS token
        k = int((atten.size(1)-2)*self.ratio)
        bs = features.size(0)
        atten[torch.arange(bs), :, text.argmax(dim=-1)] = -1 # last token 
        atten[torch.arange(bs), :, 0] = -1 # first token 
        atten = atten[torch.arange(bs), text.argmax(dim=-1), :] # 64 x 77
        atten = atten * mask
        atten = atten.masked_fill(mask == 0, -1e4)
        
        k = max(1, k)
        atten_topK_raw = atten.topk(dim=-1,k = k)[1]
        atten_topK = atten_topK_raw.unsqueeze(-1).expand(bs,k,features.size(2)) # 64 x k x 512
        features = torch.gather(input=features,dim=1,index=atten_topK)  # 64 x k x 512
        features = l2norm(features, dim=-1)

        lengths = torch.Tensor([lengths[i] if lengths[i] < k else k for i in range(bs)]) # Keep at least K
        lengths = lengths.to(features.device).long().clamp(min=1)
        valid = torch.arange(k, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        cap_emb = self.linear(features.half())
        features = self.mlp(features) + cap_emb

        logits = self.reliability_head(features).squeeze(-1).float()
        selected_atten = atten.gather(dim=1, index=atten_topK_raw).detach().float()
        logits = logits + 0.1 * selected_atten
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=1).type_as(features)
        features_weighted = (features * weights.unsqueeze(-1)).sum(dim=1)
        features_max = maxk_pool1d_var(features, 1, 1, lengths)
        features = (1.0 - self.reliability_blend) * features_max + self.reliability_blend * features_weighted
        
        return features.float()

class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024,ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.reliability_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1)
        )
        self.reliability_blend = 0.1
    
    def forward(self, base_features, atten):
        k = int((atten.size(1)-1)*self.ratio) # 192
        
        bs = base_features.size(0)
        atten[torch.arange(bs), :, 0] = -1 # CLS token   
        k = max(1, k)
        atten_topK_raw = atten[:,0].topk(dim=-1,k = k)[1]
        
        atten_topK = atten_topK_raw.unsqueeze(-1).expand(bs, k, base_features.size(2)) # 64 x k x 512
        base_features = torch.gather(input=base_features,dim=1,index=atten_topK)  # 64 x k x 512
        base_features = l2norm(base_features, dim=-1) 
        base_features = base_features.half()
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)
        
        features = self.fc(base_features)
        features = self.mlp(base_features) + features 

        logits = self.reliability_head(features).squeeze(-1).float()
        selected_atten = atten[:, 0].gather(dim=1, index=atten_topK_raw).detach().float()
        logits = logits + 0.1 * selected_atten
        weights = torch.softmax(logits, dim=1).type_as(features)
        features_weighted = (features * weights.unsqueeze(-1)).sum(dim=1)
        features_max = maxk_pool1d_var(features, 1, 1, feat_lengths)
        features = (1.0 - self.reliability_blend) * features_max + self.reliability_blend * features_weighted
 
        return features.float()
