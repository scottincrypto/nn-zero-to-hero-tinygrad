import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    # import torch
    # import tinygrad
    # import torch.nn.functional as F
    import matplotlib.pyplot as plt # for making figures
    from tinygrad import Tensor, Context, Device
    import math
    # '%matplotlib inline' command supported automatically in marimo

    # %set_env GPU=0
    # %set_env CPU=1
    print(Device.get_available_devices())
    print(Device.DEFAULT)
    Device.DEFAULT = 'CPU'
    print(Device.DEFAULT)
    return Device, Tensor, math, plt


@app.cell
def _():
    # read in all the words
    words = open('..\\..\\makemore\\names.txt', 'r').read().splitlines()
    words[:8]
    return (words,)


@app.cell
def _(words):
    len(words)
    return


@app.cell
def _(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: _i + 1 for _i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {_i: s for s, _i in stoi.items()}
    print(itos)
    return itos, stoi


@app.cell
def _(Tensor, stoi, words):
    block_size = 3
    X, Y = ([], [])
    for w in words:
        _context = [0] * block_size
        for ch in w + '.':
            _ix = stoi[ch]
            X.append(_context)
            Y.append(_ix)
            _context = _context[1:] + [_ix]
    X = Tensor(X)
    Y = Tensor(Y)
    return X, Y


@app.cell
def _(X, Y):
    X.shape, X.dtype, Y.shape, Y.dtype
    return


@app.cell
def _(Tensor, stoi, words):
    block_size_1 = 3

    def build_dataset(words):
        X, Y = ([], [])
        for w in words:
            _context = [0] * block_size_1
            for ch in w + '.':
                _ix = stoi[ch]
                X.append(_context)
                Y.append(_ix)
                _context = _context[1:] + [_ix]
        X = Tensor(X)
        Y = Tensor(Y)
        print(X.shape, Y.shape)
        return (X, Y)
    import random
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])
    return Xdev, Xtr, Ydev, Ytr, block_size_1


@app.cell
def _(Tensor):
    C = Tensor.randn((27, 2))
    return (C,)


@app.cell
def _(C, X):
    emb = C[X]
    emb.shape
    return (emb,)


@app.cell
def _(Tensor):
    W1 = Tensor.randn((6, 100))
    b1 = Tensor.randn(100)
    return W1, b1


@app.cell
def _(Tensor, W1, b1, emb):
    h = Tensor.tanh(emb.view(-1, 6) @ W1 + b1)
    return (h,)


@app.cell
def _(h):
    h.numpy()
    return


@app.cell
def _(h):
    h.shape
    return


@app.cell
def _(Tensor):
    W2 = Tensor.randn((100, 27))
    b2 = Tensor.randn(27)
    return W2, b2


@app.cell
def _(W2, b2, h):
    logits = h @ W2 + b2
    return (logits,)


@app.cell
def _(logits):
    logits.shape
    return


@app.cell
def _(logits):
    counts = logits.exp()
    return (counts,)


@app.cell
def _(counts):
    prob = counts / counts.sum(1, keepdim=True)
    return (prob,)


@app.cell
def _(prob):
    prob.shape
    return


@app.cell
def _():
    # loss = -prob[Tensor.arange(32), Y].log().mean()
    # loss
    return


@app.cell
def _():
    # ------------ now made respectable :) ---------------
    return


@app.cell
def _(Xtr, Ytr):
    Xtr.shape, Ytr.shape # dataset
    return


@app.cell
def _(Tensor):
    _g = Tensor.manual_seed(2147483647)
    C_1 = Tensor.randn((27, 10), generator=_g)
    W1_1 = Tensor.randn((30, 200), generator=_g)
    b1_1 = Tensor.randn(200, generator=_g)
    W2_1 = Tensor.randn((200, 27), generator=_g)
    b2_1 = Tensor.randn(27, generator=_g)
    parameters = [C_1, W1_1, b1_1, W2_1, b2_1]
    return C_1, W1_1, W2_1, b1_1, b2_1, parameters


@app.cell
def _(parameters):
    sum((_p.numel() for _p in parameters))
    return


@app.cell
def _(parameters):
    for _p in parameters:
        _p.requires_grad = True
    return


@app.cell
def _(Tensor):
    lre = Tensor.linspace(-3, 0, 1000)
    lrs = 10**lre
    return


@app.cell
def _():
    lri = []
    lossi = []
    stepi = []
    return lossi, stepi


@app.cell
def _(Tensor, Xtr):
    print(Tensor.randint(32, low=0, high=Xtr.shape[0]).numpy())
    return


@app.cell
def _(
    C_1,
    Device,
    Tensor,
    W1_1,
    W2_1,
    Xtr,
    Ytr,
    b1_1,
    b2_1,
    lossi,
    math,
    parameters,
    stepi,
):
    print(Device.DEFAULT)
    for _i in range(100):
        _ix = Tensor.randint(32, low=0, high=Xtr.shape[0])
        emb_1 = C_1[Xtr[_ix]]
        h_1 = Tensor.tanh(emb_1.view(-1, 30) @ W1_1 + b1_1)
        logits_1 = h_1 @ W2_1 + b2_1
        _loss = Tensor.cross_entropy(logits_1, Ytr[_ix])
        for _p in parameters:
            _p.grad = None
        _loss.backward()
        lr = 0.1 if _i < 100000 else 0.01
        for _p in parameters:
            _p.requires_grad = False
            _p.assign(_p - lr * _p.grad)
            _p.requires_grad = True
        stepi.append(_i)
        lossi.append(math.log10(_loss.item()))
    print(_loss.item())
    return


@app.cell
def _(lossi, plt, stepi):
    plt.plot(stepi, lossi)
    return


@app.cell
def _(C_1, Tensor, W1_1, W2_1, Xtr, Ytr, b1_1, b2_1):
    emb_2 = C_1[Xtr]
    h_2 = Tensor.tanh(emb_2.view(-1, 30) @ W1_1 + b1_1)
    logits_2 = h_2 @ W2_1 + b2_1
    _loss = Tensor.cross_entropy(logits_2, Ytr)
    _loss
    return


@app.cell
def _(C_1, Tensor, W1_1, W2_1, Xdev, Ydev, b1_1, b2_1):
    emb_3 = C_1[Xdev]
    h_3 = Tensor.tanh(emb_3.view(-1, 30) @ W1_1 + b1_1)
    logits_3 = h_3 @ W2_1 + b2_1
    _loss = Tensor.cross_entropy(logits_3, Ydev)
    _loss
    return


@app.cell
def _(C_1, itos, plt):
    plt.figure(figsize=(8, 8))
    plt.scatter(C_1[:, 0].data(), C_1[:, 1].data(), s=200)
    for _i in range(C_1.shape[0]):
        plt.text(C_1[_i, 0].item(), C_1[_i, 1].item(), itos[_i], ha='center', va='center', color='white')
    plt.grid('minor')
    return


@app.cell
def _():
    # training split, dev/validation split, test split
    # 80%, 10%, 10%
    return


@app.cell
def _(C_1, Tensor, block_size_1):
    _context = [0] * block_size_1
    C_1[Tensor([_context])].shape
    return


@app.cell
def _(C_1, Tensor, W1_1, W2_1, b1_1, b2_1, block_size_1, itos):
    _g = Tensor.manual_seed(2147483647 + 10)
    for _ in range(20):
        out = []
        _context = [0] * block_size_1
        while True:
            emb_4 = C_1[Tensor([_context])]
            h_4 = Tensor.tanh(emb_4.view(1, -1) @ W1_1 + b1_1)
            logits_4 = h_4 @ W2_1 + b2_1
            probs = Tensor.softmax(logits_4)
            _ix = Tensor.multinomial(probs, num_samples=1).item()
            _context = _context[1:] + [_ix]
            out.append(_ix)
            if _ix == 0:
                break
        print(''.join((itos[_i] for _i in out)))
    return


if __name__ == "__main__":
    app.run()
