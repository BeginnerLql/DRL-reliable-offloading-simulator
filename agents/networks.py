"""Shared MLP construction with stable layer ordering for saved checkpoints."""

from torch import nn


def build_hidden_layers(input_dim, widths, activation):
    """Return hidden layers and their output width, preserving initialization order."""
    activations = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
    layers = []
    for width in widths:
        layers.append(nn.Linear(input_dim, width))
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        layers.append(activations[activation]())
        input_dim = width
    return layers, input_dim
