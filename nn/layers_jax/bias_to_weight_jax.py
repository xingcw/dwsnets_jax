from typing import Optional, Tuple
from dataclasses import field

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

from nn.layers_jax.base_jax import BaseLayer, GeneralSetLayer

class SameLayer(BaseLayer):
    """Mapping bi -> Wi"""
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"
    is_output_layer: bool = False
    is_input_layer: bool = False

    def setup(self):
        super().setup()
        assert not (self.is_input_layer and self.is_output_layer)

        if self.is_input_layer:
            self.layer = GeneralSetLayer(
                in_features=self.in_features,
                out_features=self.out_features * self.out_shape[0],  # d0 * out_features
                reduction=self.reduction,
                bias=self.bias,
                n_fc_layers=self.n_fc_layers,
                num_heads=self.num_heads,
                set_layer=self.set_layer,
            )
        elif self.is_output_layer:
            self.layer = self._get_mlp(
                in_features=self.in_features * self.out_shape[-1],  # dL * in_features
                out_features=self.out_features * self.out_shape[-1],  # dL * out_features
                bias=self.bias,
            )
        else:
            # i != 0, L-1
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
        if self.is_input_layer:
            # (bs, d1, in_features)
            # (bs, d1, d0 * out_features)
            x = self.layer(x)
            # (bs, d1, d0, out_features)
            x = x.reshape(
                x.shape[0], self.out_shape[-1], self.out_shape[0], self.out_features
            )
            # (bs, d0, d1, out_features)
            x = jnp.transpose(x, (0, 2, 1, 3))

        elif self.is_output_layer:
            # (bs, dL, in_features)
            # (bs, dL * in_features)
            x = x.reshape(x.shape[0], -1)
            # (bs, dL * out_features)
            for layer in self.layer:
                x = layer(x)
            # (bs, dL, out_features)
            x = x.reshape(x.shape[0], self.out_shape[-1], self.out_features)
            # (bs, d{L-1}, dL, out_features)
            x = jnp.repeat(x[:, None, :, :], self.out_shape[0], axis=1)
        else:
            # (bs, di, in_features)
            # (bs, di, out_features)
            x = self.layer(x)
            # (bs, d{i-1}, di, out_features)
            x = jnp.repeat(x[:, None, :, :], self.out_shape[0], axis=1)
        return x

class SuccessiveLayers(BaseLayer):
    """Mapping bi -> Wj where i=j-1"""
    last_dim_is_output: bool = False
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"

    def setup(self):
        super().setup()
        if self.last_dim_is_output:
            in_features = self.in_features
            out_features = self.out_features * self.out_shape[1]  # dL * out_features
            # j=L-1, i=L-2, bi is of shape d{L-1}, Wj is of shape (d{L-1}, dL)
        else:
            # j!=L-1, bi is of shape d{i+1}, Wj is of shape (d{i+1}, d{i+2})
            in_features = self.in_features
            out_features = self.out_features  # out_features

        self.layer = GeneralSetLayer(
            in_features=in_features,
            out_features=out_features,  # dL * out_features
            reduction=self.reduction,
            bias=self.bias,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer=self.set_layer,
        )

    def __call__(self, x):
        if self.last_dim_is_output:
            # (bs, d{L-1}, in_features)
            # (bs, d{L-1}, dL * out_features)
            x = self.layer(x)
            # (bs, d{L-1}, dL * out_features)
            x = x.reshape(x.shape[0], *self.out_shape, self.out_features)
        else:
            # (bs, d{i+1}, in_features)
            # (bs, d{i+1}, out_features)
            x = self.layer(x)
            # (bs, d{i+1}, d{i+2} out_features)
            x = jnp.repeat(x[:, :, None, :], self.out_shape[-1], axis=2)
        return x

class NonNeighborInternalLayer(BaseLayer):
    """Mapping bi -> Wj where i != j,j-1"""
    last_dim_is_input: bool = False
    first_dim_is_output: bool = False
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        if self.first_dim_is_output:
            # i = L-1
            if self.last_dim_is_input:
                # i = L-1, j = 0
                in_features = self.in_features * self.in_shape[-1]  # in_features * dL
                out_features = self.out_features * self.out_shape[0]  # out_features * d0
            else:
                # i = L-1, j != 0
                in_features = self.in_features * self.in_shape[-1]  # in_features * dL
                out_features = self.out_features  # out_features
        else:
            # i != L-1
            if self.last_dim_is_input:
                # i != L-1, j = 0
                in_features = self.in_features  # in_features
                out_features = self.out_features * self.out_shape[0]  # out_features * d0
            else:
                # i != L-1, j != 0
                in_features = self.in_features  # in_features
                out_features = self.out_features  # out_features

        self.layer = self._get_mlp(
            in_features=in_features,
            out_features=out_features,
            bias=self.bias,
        )

    def __call__(self, x):
        if self.first_dim_is_output:
            # i = L-1
            if self.last_dim_is_input:
                # i = L-1, j = 0
                # (bs, dL, in_features)
                # (bs, dL * in_features)
                x = x.reshape(x.shape[0], -1)
                # (bs, d0 * out_features)
                for layer in self.layer:
                    x = layer(x)
                # (bs, d0, out_features)
                x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
                # (bs, d0, d1, out_features)
                x = jnp.repeat(x[:, :, None, :], self.out_shape[1], axis=2)
            else:
                # i = L-1, j != 0
                # (bs, dL, in_features)
                # (bs, dL * in_features)
                x = x.reshape(x.shape[0], -1)
                # (bs, out_features)
                for layer in self.layer:
                    x = layer(x)
                # (bs, d{j-1}, dj, out_features)
                x = jnp.repeat(x[:, None, None, :], self.out_shape[0], axis=1)
                x = jnp.repeat(x, self.out_shape[1], axis=2)
        else:
            # i != L-1
            if self.last_dim_is_input:
                # i != L-1, j = 0
                # (bs, d{i+1}, in_features)
                # (bs, in_features)
                x = self._reduction(x, axis=1)
                # (bs, d0 * out_features)
                for layer in self.layer:
                    x = layer(x)
                # (bs, d0, out_features)
                x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
                # (bs, d0, d1, out_features)
                x = jnp.repeat(x[:, :, None, :], self.out_shape[1], axis=2)
            else:
                # i != L-1, j != 0
                # (bs, d{i+1}, in_features)
                # (bs, in_features)
                x = self._reduction(x, axis=1)
                # (bs, out_features)
                for layer in self.layer:
                    x = layer(x)
                # (bs, d{j-1}, dj, out_features)
                x = jnp.repeat(x[:, None, None, :], self.out_shape[0], axis=1)
                x = jnp.repeat(x, self.out_shape[1], axis=2)
        return x

class BiasToWeightBlock(BaseLayer):
    """Block of layers mapping biases to weights"""
    weight_shapes: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    bias_shapes: Tuple[Tuple[int,], ...] = field(default_factory=tuple)
    diagonal: bool = False
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"

    def setup(self):
        super().setup()
        assert len(self.weight_shapes) == len(self.bias_shapes)
        self.n_layers = len(self.weight_shapes)

        # Create layers dictionary
        layers = {}
        
        # construct layers:
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and not ((i == j) or (i == j - 1)):
                    continue
                if i == j:
                    layers[f"{i}_{j}"] = SameLayer(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.bias_shapes[i],
                        out_shape=self.weight_shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        num_heads=self.num_heads,
                        set_layer=self.set_layer,
                        n_fc_layers=self.n_fc_layers,
                        is_input_layer=(
                            i == 0
                        ),  # todo: make sure this condition is correct
                        is_output_layer=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                elif i == j - 1:
                    layers[f"{i}_{j}"] = SuccessiveLayers(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.bias_shapes[i],
                        out_shape=self.weight_shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        num_heads=self.num_heads,
                        set_layer=self.set_layer,
                        n_fc_layers=self.n_fc_layers,
                        last_dim_is_output=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                else:
                    layers[f"{i}_{j}"] = NonNeighborInternalLayer(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.bias_shapes[i],
                        out_shape=self.weight_shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        last_dim_is_input=(
                            j == 0
                        ),  # todo: make sure this condition is correct
                        first_dim_is_output=(
                            i == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )

        self.layers = layers

    def __call__(self, x: Tuple[jnp.ndarray, ...]):
        # x is a tuple of tensors, one for each layer
        assert len(x) == self.n_layers
        out_weights = [
            0.0,
        ] * len(x)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and not ((i == j) or (i == j - 1)):
                    continue
                out_weights[j] = out_weights[j] + self.layers[f"{i}_{j}"](x[i])

        return tuple(out_weights)