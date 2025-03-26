import torch
import torch.nn as nn

from torch.nn import functional as F


# =======================
batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_dim = 384
heads_number = 6
layers_number = 6
dropout = 0.2
# =======================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):

        T = input.shape[1]

        key = self.key(input)
        query = self.query(input)
        value = self.value(input)

        weights = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5
        weights = F.softmax(weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')), dim=-1)
        return self.dropout(weights) @ value


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        return self.dropout(self.linear(torch.cat([head(input) for head in self.heads], dim=-1)))


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):

    def __init__(self, embedding_dim, heads_number):
        super().__init__()

        head_size = embedding_dim // heads_number

        self.self_attention = MultiHeadAttention(heads_number, head_size)
        self.feed_forward = FeedFoward(embedding_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, input):

        input = input + self.self_attention(self.layer_norm1(input))
        output = input + self.feed_forward(self.layer_norm2(input))
        return output


class Transformer(nn.Module):

    def __init__(self, vocabluary_size, embedding_dim, layers_number):
        super().__init__()

        self.token_embedding = nn.Embedding(vocabluary_size, embedding_dim)
        self.positional_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[DecoderBlock(embedding_dim, heads_number) for _ in range(layers_number)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocabluary_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input, targets=None):
        T = input.shape[-1]

        tokens_embedding = self.token_embedding(input)
        positional_embedding = self.positional_embedding(
            torch.arange(T, device=device))
        embed = tokens_embedding + positional_embedding
        out = self.blocks(embed)
        out = self.layer_norm(out)
        logits = self.linear(out)
        
        prob_fn = F.softmax

        return prob_fn(logits)

    def generate(self, input, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = input[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input = torch.cat((input, idx_next), dim=1)
        return input

