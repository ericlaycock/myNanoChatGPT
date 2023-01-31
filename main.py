# import wget
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# file = wget.download(url);

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

#tokenize chars

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# #better tokenizing process: tiktoken
# #pip install tiktoken
# import tiktoken
# enc = tiktoken.get_encoding(text)
# assert enc.decode.enc.encode(text) == text


#let's now encode entre dataset and store it in a torch.Tensor
import torch
data = torch.tensor(encode(text),dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# print(train_data[0:1000])

torch.manual_seed(1337)
batch_size = 4 #how many independent sequences will be process in parallel
block_size = 8 #maximum context length for predictions / length of each batch

def get_batch(split):
    #generate small batch of data of inputs x and targets/labels y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #randomize offset
    x = torch.stack([data[i:i+block_size] for i in ix]) #inputs
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #targets, offset by +1
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size): #across batches
    for t in range(block_size): #across the batch itself
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when the input is {context.tolist()} the target is: {target}")

import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)