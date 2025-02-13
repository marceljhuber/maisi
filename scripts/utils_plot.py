# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import AsDiscrete


def find_label_center_loc(image):
    """Find center location for 2D image."""
    return [shape // 2 for shape in image.shape[-2:]]


def normalize_label_to_uint8(colorize, label, n_label):
    """Normalize and colorize a 2D label tensor to uint8."""
    with torch.no_grad():
        post_label = AsDiscrete(to_onehot=n_label)
        label = post_label(label).permute(1, 0, 2)
        label = F.conv2d(label, weight=colorize)
        label = torch.clip(label, 0, 1).squeeze().permute(1, 2, 0).cpu().numpy()
    return (label * 255).astype(np.uint8)


def visualize_2d(image):
    """Visualize 2D image.

    Args:
        image: Tensor of shape [C, H, W] or [1, C, H, W]
    Returns:
        2D numpy array of shape [H, W]
    """
    # If it's a torch tensor, get numpy array
    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()

    # Remove batch dimension if present
    if len(image.shape) == 4:
        image = image.squeeze(0)

    # Remove channel dimension
    if len(image.shape) == 3:
        image = image.squeeze(0)

    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2

    return image  # Should now be [H, W]


def show_image(image, title="mask"):
    """
    Plot and display an input image.

    Args:
        image (numpy.ndarray): Image to be displayed. Expected shape: [H, W] for grayscale or [H, W, 3] for RGB.
        title (str, optional): Title for the plot. Defaults to "mask".
    """
    plt.figure("check", (24, 12))
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.imshow(image)
    plt.show()


def to_shape(a, shape):
    """Pad 2D array to desired shape."""
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )


def get_xyz_plot(
    image, center_loc_axis=None, mask_bool=False, n_label=None, colorize=False
):
    """Create visualization of 2D image."""
    img = visualize_2d(image)
    if mask_bool:
        img = img > 0.5
    return torch.from_numpy(img)
