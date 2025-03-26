from transformer_decoder import Transformer

import torch
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from tqdm.auto import tqdm as tqdma
from torch.nn import functional as F

from dataset import PoemDataset

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

text = pd.read_csv(
    "/content/data.csv")["text"].to_list()
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer.model_max_length = 1000000000


train_dataset = PoemDataset(
    "".join(text[:int(0.9*len(text))]), tokenizer, block_size)
train_dataloader = iter(DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False))


test_dataset = PoemDataset(
    "".join(text[int(0.9*len(text)):]), tokenizer, block_size)
test_dataloader = iter(DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False))


@torch.no_grad()
def loss_estim():
    model.eval()
    train_losses = torch.zeros(eval_iters)
    test_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x_train, y_train = next(train_dataloader)
        x_test, y_test = next(test_dataloader)

        logits_train = model(x_train)
        logits_test = model(x_test)

        B, T, C = logits_train.shape

        logits_train = logits_train.view(B*T, C)
        logits_test = logits_test.view(B*T, C)

        y_train = y_train.view(B*T)
        y_test = y_test.view(B*T)

        train_losses[k] = F.cross_entropy(logits_train, y_train)
        test_losses[k] = F.cross_entropy(logits_test, y_test)

    return {"train": train_losses.mean(), "test": test_losses.mean()}

model = Transformer(tokenizer.vocab_size, embedding_dim, layers_number).to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

progress_bar = tqdma(total=max_iters)

for itera, (x, y) in zip(range(max_iters), train_dataloader):
    if itera % eval_interval == 0 or itera == max_iters - 1:
        losses = loss_estim()
        print(f"step {itera}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    progress_bar.update(1)

progress_bar.close()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

