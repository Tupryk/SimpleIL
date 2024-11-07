import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


dataset_size = 50

# Get data from a circle
thetas = np.random.uniform(0, 2*np.pi, dataset_size)
x = np.cos(thetas) + np.random.normal(0, 3e-2, dataset_size)
y = np.sin(thetas) + np.random.normal(0, 3e-2, dataset_size)
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


def train(nepochs: int = 10):
  model = Net()
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  losses = []
  for epoch in range(nepochs):
    for [data] in loader:
      data = data.to(device)
      optimizer.zero_grad()

      # Fwd pass
      t = torch.rand(size=(data.shape[0], 1))
      noise = torch.randn(*data.shape, device=device)
      model_in = data * t + noise * ( torch.ones(size=(data.shape[0], 1)) - t )
      out = model(model_in, t.to(device))
      loss = torch.mean((noise - out)**2)
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
  x_t = torch.randn((n_samples, 2)).to(device)
 
  for _ in range(n_steps):
    x_t += torch.randn((n_samples, 2)).to(device) * .001

    t = torch.ones((n_samples, 1)).to(device)

    noise_prediction = trained_model(x_t, t)
    noise_prediction /= n_steps
    x_t -= noise_prediction

  return x_t

## Run training
trained_model = train(20_000)

samples = sample(n_samples=1000).detach().cpu().numpy()
print(samples)

plt.figure(figsize=(5,5))
plt.scatter(*(samples.T))
plt.show()
