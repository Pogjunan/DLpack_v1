## import ##

import os
import imageio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections.abc import Iterable


##function ##

def truncate(value, max_length=200):
    """Truncate string representation of a value if it exceeds max_length."""
    value_str = str(value)
    return value_str if len(value_str) <= max_length else value_str[:max_length] + "..."

def display_items(items, max_head=5, max_tail=5):
    """Display head and tail of items, skipping the middle if necessary."""
    if not hasattr(items, "__len__"):
        print(f"Type: {type(items).__name__}, Value: {truncate(items)}")
        return
    
    total_items = len(items)
    print(f"Total items: {total_items}")
    
    for idx, item in enumerate(items):
        if idx >= max_head and idx < total_items - max_tail:
            if idx == max_head:
                print("...")
            continue
        print(f"[{idx}] Type: {type(item).__name__}, Value: {truncate(item)}")

def show(item, max_depth=2, max_head=5, max_tail=5):
    """
    Display structured overview of an item.
    Supports dictionaries, lists, or any iterable.
    """
    if isinstance(item, dict):
        print(f"Dictionary Overview: {len(item)} keys")
        for idx, (key, value) in enumerate(item.items()):
            print(f"[{idx}] Key: {key}, Type: {type(value).__name__}, Value: {truncate(value)}")
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        print("List Overview:")
        display_items(item, max_head=max_head, max_tail=max_tail)
    else:
        print(f"Unsupported Type: {type(item).__name__}, Value: {truncate(item)}")



import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def step(self, closure=None):
        assert closure is not None, "SAM optimizer requires a closure function."
        closure = torch.enable_grad()(closure)

        # 첫 번째 순전파 및 역전파
        loss = closure()  # loss 계산 및 backward 수행

        # Step 1: 파라미터를 업데이트하여 경사면을 따라 이동
        self.first_step()

        # 두 번째 순전파 및 역전파 (그래디언트 계산 필요 없음)
        with torch.no_grad():
            closure()

        # Step 2: 원래 파라미터로 복원하고 실제 업데이트 수행
        self.second_step()

        return loss  # 첫 번째 loss 반환

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # 원래 파라미터로 복원

        self.base_optimizer.step()  # 실제 업데이트 수행

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

def image_animation(image_dir, output_gif, duration=0.7):
    """
    Create an animated GIF from a series of images in a directory.

    Args:
        image_dir (str or Path): Directory containing the image files.
        output_gif (str): Filename for the output GIF.
        duration (float): Duration for each frame in the GIF (in seconds).
    """
    # image_dir = "image_dir"  # Directory containing PNG images
    # output_gif = "2nd_trial.gif"  # Output GIF filename
    # image_animation(image_dir, output_gif, duration=0.7)
    
    # Convert image_dir to Path object if it's a string
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.png"), key=lambda x: x.name)
    if not image_files:
        print("No image files found in the specified directory.")
        return

    with imageio.get_writer(output_gif, mode="I", duration=duration) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Animation saved as {output_gif}")



