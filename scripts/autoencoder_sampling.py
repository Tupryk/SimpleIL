import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device Name: {torch.cuda.get_device_name(device)}" if device.type == "cuda" else "CPU")

model = Autoencoder().to(device)
model.load_state_dict(torch.load('./models/autoencoder_sota.pth'))

data = torch.Tensor(np.load("./data/arrays.npz")["arr_0"]).to(device)
print(data.shape)
y = model.encode(data)
print(y.shape)
print(y.min())
print(y.max())

tensor = (torch.rand(16, 6).to(device) * (y.max()-y.min())) + y.min()
print(tensor)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

# Generate and plot each image
images = model.decode(tensor)
print(images.shape)
images = images.detach().cpu().numpy()
images = np.maximum(images, 0)
images = np.minimum(images, 1)
print(images.shape)
for i in range(4):
    for j in range(4):
        image = np.transpose(images[i*4+j], (1, 2, 0))
        axes[i, j].imshow(image)
        axes[i, j].axis('off')  # Hide the axis

plt.tight_layout()
plt.show()
