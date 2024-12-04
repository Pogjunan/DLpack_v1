import numpy as np
import pandas as pd
from collections.abc import Iterable

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