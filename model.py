import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self):
        pass


class LayerNorm(nn.Module):

    def __init__(self):
        pass


class GELU(nn.Module):
    """
    GELU implementation as described in the corresponding publication:
    https://arxiv.org/pdf/1606.08415
    """

    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):

    def __init__(self):
        pass


class LayerNorm(nn.Module):

    def __init__(self):
        pass


class TransformerBlock(nn.Module):

    def __init__(self):
        pass


class GPT(nn.Module):

    def __init__(self):
        pass
