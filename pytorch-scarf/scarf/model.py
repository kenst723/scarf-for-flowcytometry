import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.uniform import Uniform
import math

class PLREncoding(nn.Module):
    def __init__(self, num_features: int, num_frequencies: int, sigma: float, embedding_dim: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_frequencies = num_frequencies
        self.embedding_dim = embedding_dim
        
        # 1. Periodic (特徴量ごとに独立した周波数: num_features x num_frequencies)
        frequencies = torch.randn(num_features, num_frequencies) * sigma
        self.frequencies = nn.Parameter(frequencies)
        
        # 2. Linear (特徴量ごとに独立した全結合層)
        # 入力次元は 2 * num_frequencies, 出力次元は embedding_dim
        self.linear_weights = nn.Parameter(torch.Tensor(num_features, 2 * num_frequencies, embedding_dim))
        self.linear_bias = nn.Parameter(torch.Tensor(num_features, embedding_dim))
        
        # Pytorch標準のLinear層と同じ方法で初期化
        nn.init.kaiming_uniform_(self.linear_weights, a=math.sqrt(5))
        fan_in = 2 * num_frequencies
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.linear_bias, -bound, bound)
        
        # 3. ReLU
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, num_features)
        
        # --- 1. Periodic ---
        # v: (batch_size, num_features, num_frequencies)
        v = 2 * math.pi * x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        # periodic: (batch_size, num_features, 2 * num_frequencies)
        periodic = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        
        # --- 2. Linear ---
        # out: (batch_size, num_features, embedding_dim)
        out = torch.einsum('bfj,fjd->bfd', periodic, self.linear_weights) + self.linear_bias
        
        # --- 3. ReLU ---
        out = self.relu(out)
        
        # 平坦化: (batch_size, num_features * embedding_dim)
        return out.flatten(1)


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden: int, dropout: float = 0.0) -> None:
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_low: int,
        features_high: int,
        dim_hidden_encoder: int,
        num_hidden_encoder: int,
        dim_hidden_head: int,
        num_hidden_head: int,
        corruption_rate: float = 0.6,
        dropout: float = 0.0,
        num_frequencies: int = 4,
        sigma: float = 1.0,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        self.periodic = PLREncoding(input_dim, num_frequencies, sigma, embedding_dim)
        expanded_input_dim = input_dim * embedding_dim

        self.encoder = MLP(expanded_input_dim, dim_hidden_encoder, num_hidden_encoder, dropout)
        self.pretraining_head = MLP(dim_hidden_encoder, dim_hidden_head, num_hidden_head, dropout)

        # uniform disstribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_rate = corruption_rate

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _ = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true
        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        encoded_x = self.encoder(self.periodic(x))
        encoded_corrupted = self.encoder(self.periodic(x_corrupted))
        
        embeddings = self.pretraining_head(encoded_x)
        embeddings_corrupted = self.pretraining_head(encoded_corrupted)

        return embeddings, embeddings_corrupted

    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(self.periodic(x))
