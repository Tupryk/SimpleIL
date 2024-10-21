"""
Simple Diffusion Example to learn to draw samples from a circle.

Written by Cornelius Braun, TU Berlin.
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Get data from a circle
thetas = np.random.uniform(0, 2*np.pi, 50)
x = np.cos(thetas) + np.random.normal(0, 3e-2, 50)
y = np.sin(thetas) + np.random.normal(0, 3e-2, 50)
data = np.vstack((x, y)).T
print(data)
plt.figure(figsize=(5, 5))
plt.scatter(x, y)

print(data.shape)

# Define dataset
device = "cpu"
dataset = torch.utils.data.TensorDataset(torch.Tensor(data))
loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

# Define net
class Net(nn.Module):
  def __init__(self, nhidden: int = 256):
    super().__init__()
    layers = [nn.Linear(3, nhidden)]
    for _ in range(2):
      layers.append(nn.Linear(nhidden, nhidden))
    layers.append(nn.Linear(nhidden, 2))
    self.linears = nn.ModuleList(layers)

    # init using kaiming
    for layer in self.linears:
      nn.init.kaiming_uniform_(layer.weight)

  def forward(self, x, t):
    x = torch.concat([x, t], axis=-1)
    for l in self.linears[:-1]:
      x = nn.ReLU()(l(x))
    return self.linears[-1](x)


def get_alpha_betas(N: int):
  """Schedule from the original paper.
  """
  beta_min = 0.1
  beta_max = 20.
  betas = np.array([beta_min/N + i/(N*(N-1))*(beta_max-beta_min) for i in range(N)])
  alpha_bars = np.cumprod(1 - betas)
  return alpha_bars, betas

a, b = get_alpha_betas(100)
# plt.plot(a, label="Amount Signal")
# plt.plot(1 - a, label="Amount Noise")
# plt.legend()


def train(nepochs: int = 10, denoising_steps: int = 100):
  """Alg 1 from the DDPM paper"""
  model = Net()
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  alpha_bars, _ = get_alpha_betas(denoising_steps)      # Precompute alphas
  losses = []
  for epoch in range(nepochs):
    for [data] in loader:
      data = data.to(device)
      optimizer.zero_grad()

      # Fwd pass
      t = torch.randint(denoising_steps, size=(data.shape[0],))  # sample timesteps - 1 per datapoint
      alpha_t = torch.index_select(torch.Tensor(alpha_bars), 0, t).unsqueeze(1).to(device)    # Get the alphas for each timestep
      noise = torch.randn(*data.shape, device=device)   # Sample DIFFERENT random noise for each datapoint
      model_in = alpha_t**.5 * data + noise*(1-alpha_t)**.5   # Noise corrupt the data (eq14)
      out = model(model_in, t.unsqueeze(1).to(device))
      loss = torch.mean((noise - out)**2)     # Compute loss on prediction (eq14)
      losses.append(loss.detach().cpu().numpy())

      # Bwd pass
      loss.backward()
      optimizer.step()

    if (epoch+1) % 5_000 == 0:
        mean_loss = np.mean(np.array(losses))
        losses = []
        print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))

  return model


def sample(n_samples: int = 50, n_steps: int=100):
  """Alg 2 from the DDPM paper."""
  x_t = torch.randn((n_samples, 2)).to(device)
  alpha_bars, betas = get_alpha_betas(n_steps)
  alphas = 1 - betas
  for t in range(len(alphas))[::-1]:
    ts = t * torch.ones((n_samples, 1)).to(device)
    ab_t = alpha_bars[t] * torch.ones((n_samples, 1)).to(device)  # Tile the alpha to the number of samples
    z = (torch.randn((n_samples, 2)) if t > 1 else torch.zeros((n_samples, 2))).to(device)
    model_prediction = trained_model(x_t, ts)
    x_t = 1 / alphas[t]**.5 * (x_t - betas[t]/(1-ab_t)**.5 * model_prediction)
    x_t += betas[t]**0.5 * z

  return x_t

## Run training
trained_model = train(20_000)

samples = sample().detach().cpu().numpy()
samples.shape

plt.figure(figsize=(5,5))
plt.scatter(*(samples.T))
plt.show()
