import math
from typing import Optional, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import numpy as np

# Note: We'll need to implement our own positional encoding since we can't use the PyTorch version
class PositionalEncoding(nn.Module):
    sigma: float = 10.0
    m: int = 10  # number of frequencies

    @nn.compact
    def __call__(self, x):
        # Create frequency bands
        freq_bands = jnp.linspace(0.0, self.m - 1, self.m)
        # Expand dimensions for broadcasting
        x_expanded = x[..., None]  # [..., input_dim, 1]
        # Compute the encoding
        proj = (2.0 * jnp.pi * self.sigma ** (freq_bands / self.m)) * x_expanded  # [..., input_dim, m]
        # Compute sin and cos
        sin_proj = jnp.sin(proj)  # [..., input_dim, m]
        cos_proj = jnp.cos(proj)  # [..., input_dim, m]
        # Concatenate and reshape
        encoded = jnp.concatenate([sin_proj, cos_proj], axis=-1)  # [..., input_dim, 2*m]
        return encoded.reshape(*x.shape[:-1], -1)  # [..., input_dim * 2*m]

class GaussianEncoding(nn.Module):
    sigma: float = 10.0
    input_size: int = 2
    encoded_size: int = 10

    def setup(self):
        # Initialize random projection matrix
        key = self.make_rng('params')
        self.projection = random.normal(key, (self.input_size, self.encoded_size)) * self.sigma

    def __call__(self, x):
        # Project input
        proj = 2 * jnp.pi * jnp.matmul(x, self.projection)  # [..., encoded_size]
        # Compute sin and cos
        sin_proj = jnp.sin(proj)  # [..., encoded_size]
        cos_proj = jnp.cos(proj)  # [..., encoded_size]
        # Concatenate
        return jnp.concatenate([sin_proj, cos_proj], axis=-1)  # [..., 2*encoded_size]

class Siren(nn.Module):
    dim_in: int
    dim_out: int
    w0: float = 30.0
    c: float = 6.0
    is_first: bool = False
    use_bias: bool = True
    activation: Optional[Callable] = None

    def setup(self):
        # Initialize weights and bias
        w_std = (1 / self.dim_in) if self.is_first else (jnp.sqrt(self.c / self.dim_in) / self.w0)
        
        # Initialize weights
        self.weight = self.param('weight', 
                               lambda key: random.uniform(key, 
                                                        (self.dim_out, self.dim_in),
                                                        minval=-w_std,
                                                        maxval=w_std))
        # use this for testing
        # self.weight = self.param('weight', 
        #                         lambda key: jnp.ones((self.dim_out, self.dim_in)) * w_std)
        
        # Initialize bias if needed
        if self.use_bias:
            self.bias = self.param('bias',
                                 lambda key: jnp.zeros(self.dim_out))
        else:
            self.bias = None

    def __call__(self, x):
        out = x @ self.weight.T  # Note: JAX uses different convention for matrix multiplication
        if self.bias is not None:
            out = out + self.bias
        # Apply activation
        if self.activation is None:
            return jnp.sin(self.w0 * out)
        else:
            return self.activation(out)

class INR(nn.Module):
    in_dim: int = 2
    n_layers: int = 3
    up_scale: int = 4
    out_channels: int = 1
    pe_features: Optional[int] = None
    fix_pe: bool = True

    def setup(self):
        hidden_dim = self.in_dim * self.up_scale

        # Create layers list
        layers = []
        
        # Add positional encoding if specified
        if self.pe_features is not None:
            if self.fix_pe:
                layers.append(PositionalEncoding(sigma=10.0, m=self.pe_features))
                encoded_dim = self.in_dim * self.pe_features * 2
            else:
                layers.append(GaussianEncoding(sigma=10.0, 
                                            input_size=self.in_dim, 
                                            encoded_size=self.pe_features))
                encoded_dim = self.pe_features * 2
            layers.append(Siren(dim_in=encoded_dim, dim_out=hidden_dim))
        else:
            layers.append(Siren(dim_in=self.in_dim, dim_out=hidden_dim))

        # Add hidden layers
        for _ in range(self.n_layers - 2):
            layers.append(Siren(dim_in=hidden_dim, dim_out=hidden_dim))

        # Add output layer
        layers.append(nn.Dense(
            self.out_channels,
            # kernel_init=nn.initializers.ones,  # use this for testing
            # bias_init=nn.initializers.zeros,   # use this for testing
        ))

        # Create sequential module
        self.layers = layers

    def __call__(self, x):
        # Apply layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x + 0.5 