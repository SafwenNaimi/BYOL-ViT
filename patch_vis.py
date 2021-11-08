#%%
from ctypes import cdll
cdll.LoadLibrary("libstdc++.so.6") 
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from timm import create_model

IMG_SIZE = (96, 96)
#NORMALIZE_MEAN = (0.5, 0.5, 0.5)
#NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
              T.Resize(IMG_SIZE),
              T.ToTensor(),              
              ]

transforms = T.Compose(transforms)


img = PIL.Image.open('Python Scripts/VIT/24.png')
img = img.convert('RGB')
#img_tensor = transforms(img).unsqueeze(0).to(device)

fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of Patches", fontsize=24)
fig.add_axes()
img = np.asarray(img)
for i in range(0, 36):
    x = i % 6
    y = i // 6
    patch = img[y*16:(y+1)*16, x*16:(x+1)*16]
    ax = fig.add_subplot(6, 6, i+1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(patch)
# %%
