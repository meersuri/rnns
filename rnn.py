import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RNN(torch.nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.dim = dim
        self.input_proj = torch.nn.Linear(input_dim, dim)
        self.hidden_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, input_dim)
        self.hidden_state = torch.nn.Parameter(data=torch.randn(dim))

    def forward(self, x): # B, input_dim
        next_hidden = torch.tanh(self.input_proj(x) + self.hidden_proj(self.hidden_state))
        y = self.out_proj(next_hidden)
        self.hidden_state.data = torch.mean(next_hidden, dim=[0,1])
        return y


class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.r1 = RNN(input_dim, hidden_dim)
        self.linear1 = torch.nn.Linear(input_dim, input_dim)

    def forward(self, x, y=None):
        out = self.linear1(self.r1(x))
        if y is None:
            loss = None
        else:
            loss = torch.mean(torch.mean((out - y)**2, dim=2))
        return out, loss

def gen_data(count=1000):
    pts = torch.linspace(-10, 10, count)
    x = torch.sin(pts)
    x = torch.unsqueeze(x, 1)
    return x

def get_batch(data, batch_size, block_size):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size, ))
    x = torch.stack([data[idx: idx + block_size] for idx in starts])
    y = torch.stack([data[idx + 1: idx + 1 + block_size] for idx in starts])
    return (x, y)

def plot_samples(model, device, count=100):
    x, y = get_batch(data, 1, count)
    x, y = x.to(device), y.to(device)
    yh, loss = model(x, y)
    yh = yh.detach().cpu()
    y = y.detach().cpu()
    plt.plot(y[0])
    plt.plot(yh[0])
    plt.show()


if __name__ == '__main__':
    input_dim = 1
    hidden_dim = 1
    iters = 4000
    batch_size = 4
    block_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(input_dim, hidden_dim)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = gen_data()
    plot_samples(model, device, 500)
    for i in range(iters):
        x, y = get_batch(data, batch_size, block_size)
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        yh, loss = model(x, y)
        loss.backward()
        optim.step()
        if i % 100 == 0:
            print(loss)
    plot_samples(model, device, 500)

