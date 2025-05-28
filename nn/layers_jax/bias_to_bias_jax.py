from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from dataclasses import field

from nn.layers_jax.base_jax import BaseLayer, GeneralSetLayer

class SelfToSelfLayer(BaseLayer):
    """Mapping bi -> bi"""
    is_output_layer: bool = False
    reduction: str = "max"
    bias: bool = True
    num_heads: int = 8
    set_layer: str = "sab"
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        if self.is_output_layer:
            # i=L-1
            assert self.in_shape == self.out_shape
            self.layer = self._get_mlp(
                in_features=self.in_shape[0] * self.in_features,
                out_features=self.in_shape[0] * self.out_features,
                bias=self.bias,
            )
        else:
            self.layer = GeneralSetLayer(
                in_features=self.in_features,
                out_features=self.out_features,
                reduction=self.reduction,
                bias=self.bias,
                n_fc_layers=self.n_fc_layers,
                num_heads=self.num_heads,
                set_layer=self.set_layer,
            )

    def __call__(self, x):
        # (bs, d{i+1}, in_features)
        if self.is_output_layer:
            # (bs, d{i+1} * out_features)
            x = x.reshape(x.shape[0], -1)  # flatten
            for layer in self.layer:
                x = layer(x)
            # (bs, d{i+1}, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
        else:
            # (bs, d{i+1}, out_features)
            x = self.layer(x)
        return x

class SelfToOtherLayer(BaseLayer):
    """Mapping bi -> bj"""
    first_dim_is_output: bool = False
    last_dim_is_output: bool = False
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        assert not (self.first_dim_is_output and self.last_dim_is_output)

        if self.first_dim_is_output:
            # b{L-1} -> bj
            self.layer = self._get_mlp(
                in_features=self.in_features * self.in_shape[0],  # in_features * dL
                out_features=self.out_features,
                bias=self.bias,
            )
        elif self.last_dim_is_output:
            # bi -> b{L-1}
            self.layer = self._get_mlp(
                in_features=self.in_features,
                out_features=self.out_features * self.out_shape[0],  # out_features * dL
                bias=self.bias,
            )
        else:
            # i,j != L-1
            self.layer = self._get_mlp(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=self.bias,
            )

    def __call__(self, x):
        if self.first_dim_is_output:
            # b{L-1} -> bj
            # (bs, dL, in_features)
            # (bs, dL * in_features)
            x = x.reshape(x.shape[0], -1)
            # (bs, out_features)
            for layer in self.layer:
                x = layer(x)
            # (bs, b{j+1}, out_features)
            x = jnp.repeat(x[:, None, :], self.out_shape[0], axis=1)

        elif self.last_dim_is_output:
            # bi -> b{L-1}
            # (bs, d{i+1}, in_features)
            # (bs, in_features)
            x = self._reduction(x, axis=1)
            # (bs, dL * out_features)
            for layer in self.layer:
                x = layer(x)
            # (bs, dL, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
        else:
            # i,j != L-1
            # (bs, d{i+1}, in_features)
            # (bs, in_features)
            x = self._reduction(x, axis=1)
            # (bs, out_features)
            for layer in self.layer:
                x = layer(x)
            # (bs, b{j+1}, out_features)
            x = jnp.repeat(x[:, None, :], self.out_shape[0], axis=1)

        return x

class BiasToBiasBlock(BaseLayer):
    """Block for mapping between biases"""
    shapes: Tuple[Tuple[int, ...], ...] = field(default_factory=tuple)
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"
    diagonal: bool = False

    def setup(self):
        super().setup()
        assert all([len(shape) == 1 for shape in self.shapes])
        self.n_layers = len(self.shapes)

        # Create layers dictionary
        layers = {}
        
        # construct layers:
        if self.diagonal:
            for i in range(self.n_layers):
                layers[f"{i}_{i}"] = SelfToSelfLayer(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    in_shape=self.shapes[i],
                    out_shape=self.shapes[i],
                    reduction=self.reduction,
                    bias=self.bias,
                    num_heads=self.num_heads,
                    set_layer=self.set_layer,
                    n_fc_layers=self.n_fc_layers,
                    is_output_layer=(i == self.n_layers - 1),
                )
        # full DWS layers:
        else:
            for i in range(self.n_layers):
                for j in range(self.n_layers):
                    if i == j:
                        layers[f"{i}_{j}"] = SelfToSelfLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            is_output_layer=(j == self.n_layers - 1),
                        )
                    else:
                        layers[f"{i}_{j}"] = SelfToOtherLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            n_fc_layers=self.n_fc_layers,
                            first_dim_is_output=(i == self.n_layers - 1),
                            last_dim_is_output=(j == self.n_layers - 1),
                        )

        self.layers = layers

    def __call__(self, x: Tuple[jnp.ndarray, ...]):
        # x is a tuple of tensors, one for each layer
        assert len(x) == self.n_layers
        out_biases = [
            0.0,
        ] * len(x)
        if self.diagonal:
            for i in range(self.n_layers):
                out_biases[i] = self.layers[f"{i}_{i}"](x[i])
        else:
            for i in range(self.n_layers):
                for j in range(self.n_layers):
                    out_biases[j] = out_biases[j] + self.layers[f"{i}_{j}"](x[i])

        return tuple(out_biases)