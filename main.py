# import wget
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# file = wget.download(url);

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

print(len(text))

chars = sorted(list(set(text)))
charbulary_size = len(chars)

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