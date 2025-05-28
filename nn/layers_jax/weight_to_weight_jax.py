from typing import Tuple
from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np

from nn.layers_jax.base_jax import BaseLayer, GeneralSetLayer

class GeneralMatrixSetLayer(BaseLayer):
    """General matrix set layer."""
    first_dim_is_input: bool = False
    last_dim_is_input: bool = False
    first_dim_is_output: bool = False
    last_dim_is_output: bool = False
    in_index: int = 0
    out_index: int = 0
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"

    def setup(self):
        super().setup()
        assert not (self.first_dim_is_input and self.last_dim_is_input)
        assert not (self.first_dim_is_input and self.last_dim_is_output)
        assert not (self.last_dim_is_input and self.first_dim_is_output)

        if self.in_index == self.out_index:
            assert not (self.first_dim_is_input and self.last_dim_is_input)
            self.feature_index = (
                0 if self.first_dim_is_input else 1
            )  # 0 means we are at first layer, 1 means last layer
            # this is the case we map W_i to W_i where W_i is the first or last layer's weight matrix
            in_features = self.in_features * self.in_shape[self.feature_index]
            out_features = self.out_features * self.in_shape[self.feature_index]

        elif self.in_index == self.out_index - 1:
            # this is the case we map W_i to W_j where i=j-1
            assert not (self.first_dim_is_input and self.last_dim_is_output)
            if self.first_dim_is_input:
                # i=0 and j=1
                self.feature_index = 0
                in_features = self.in_features * self.in_shape[self.feature_index]
                out_features = self.out_features
            elif self.last_dim_is_output:
                # i=L-2 and j=L-1
                self.feature_index = 1
                in_features = self.in_features
                out_features = self.out_features * self.out_shape[self.feature_index]
            else:
                # internal layers
                in_features = self.in_features
                out_features = self.out_features

        else:
            # i = j + 1
            assert self.in_index == self.out_index + 1  # in_shape[0] == out_shape[-1]
            assert not (self.last_dim_is_input and self.first_dim_is_output)
            if self.last_dim_is_input:
                # j=0, i=1
                self.feature_index = 0
                in_features = self.in_features
                out_features = self.out_features * self.out_shape[self.feature_index]

            elif self.first_dim_is_output:
                # j=L-2, i=L-1
                self.feature_index = 1
                in_features = self.in_features * self.in_shape[self.feature_index]
                out_features = self.out_features

            else:
                # internal layers
                in_features = self.in_features
                out_features = self.out_features

        self.general_set_layer = GeneralSetLayer(
            in_features=in_features,
            out_features=out_features,
            reduction=self.reduction,
            bias=self.bias,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer=self.set_layer,
        )

    def __call__(self, x):
        if self.in_index == self.out_index:
            # this is the case we map W_i to W_i where W_i is the first or last layer's weight matrix
            if self.first_dim_is_input:
                # first layer, feature_index is d0
                # (bs, d1, d0, in_features)
                x = jnp.transpose(x, (0, 2, 1, 3))

            # (bs, set_dim, feature_dim * in_features)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            # (bs, set_dim, feature_dim * out_features)
            x = self.general_set_layer(x)
            # (bs, set_dim, feature_dim, out_features)
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.in_shape[self.feature_index],
                self.out_features,
            )

            if self.first_dim_is_input:
                # permute back to (bs, d0, d1, out_features)
                x = jnp.transpose(x, (0, 2, 1, 3))

        elif (
            self.in_index == self.out_index - 1
        ):  # self.in_shape[-1] == self.out_shape[0]:
            # i -> j  where i=j-1
            if self.first_dim_is_input:
                # i=0 and j=1
                # (bs, d1, d0 * in_features)
                x = jnp.transpose(x, (0, 2, 1, 3))
                x = x.reshape(x.shape[0], x.shape[1], -1)
                # (bs, d1, out_features)
                x = self.general_set_layer(x)
                # (bs, d1, d2, out_features)
                x = jnp.expand_dims(x, 2)
                x = jnp.repeat(x, self.out_shape[-1], axis=2)

            elif self.last_dim_is_output:
                # i=L-2 and j=L-1
                # (bs, d_{L-2}, d_{L-1}, in_features)
                # (bs, d_{L-1}, in_features)
                x = self._reduction(x, axis=1)
                # (bs, d_{L-1}, d_L * out_features)
                x = self.general_set_layer(x)
                # (bs, d_{L-1}, d_L, out_features)
                x = x.reshape(x.shape[0], *self.out_shape, self.out_features)
            else:
                # internal layers
                # (bs, d_i, d_{i+1}, in_features)
                # (bs, d_{i+1}, in_features)
                x = self._reduction(x, axis=1)
                # (bs, d_{i+1}, out_features)
                x = self.general_set_layer(x)
                # (bs, d_{i+1}, d_{i+2}, out_features)
                x = jnp.expand_dims(x, 2)
                x = jnp.repeat(x, self.out_shape[-1], axis=2)

        else:
            # i = j + 1
            if self.last_dim_is_input:
                # i=1, j=0
                # (bs, d1, d2, in_features)
                # (bs, d1, in_features)
                x = self._reduction(x, axis=2)
                # (bs, d1, d0 * out_features)
                x = self.general_set_layer(x)
                # (bs, d1, d0, out_features)
                x = x.reshape(
                    x.shape[0], x.shape[1], self.out_shape[0], self.out_features
                )
                # (bs, d0, d1, out_features)
                x = jnp.transpose(x, (0, 2, 1, 3))

            elif self.first_dim_is_output:
                # i=L-1, j=L-2
                # (bs, d_{L-1}, d_L, in_features)
                # (bs, d_{L-1}, out_features)
                x = self.general_set_layer(x.reshape(x.shape[0], x.shape[1], -1))
                x = jnp.expand_dims(x, 1)
                x = jnp.repeat(x, self.out_shape[0], axis=1)

            else:
                # internal layers (j = i-1):
                # (bs, d_i, d_{i+1}, in_feature) -> (bs, d_{i-1}, d_i, out_features)
                # (bs, d_i, in_feature)
                x = self._reduction(x, axis=2)
                # (bs, d_i, out_feature)
                x = self.general_set_layer(x)
                # (bs, d_{i-1}, d_i, out_feature)
                x = jnp.expand_dims(x, 1)
                x = jnp.repeat(x, self.out_shape[0], axis=1)
        return x

class SetKroneckerSetLayer(BaseLayer):
    """Set Kronecker set layer."""
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        # todo: bias is overparametrized here. we can reduce the number of parameters
        self.d1, self.d2 = self.in_shape

        self.lin_all = self._get_mlp(self.in_features, self.out_features, bias=self.bias)
        self.lin_n = self._get_mlp(self.in_features, self.out_features, bias=self.bias)
        self.lin_m = self._get_mlp(self.in_features, self.out_features, bias=self.bias)
        self.lin_both = self._get_mlp(self.in_features, self.out_features, bias=self.bias)

        # todo: add attention support
        # if reduction == "attn":
        #     self.attn0 = Attn(self.d2 * self.in_features)
        #     self.attn1 = Attn(self.d1 * self.in_features)
        #     self.attn2 = Attn(self.in_features)

    def __call__(self, x):
        # x is [b, d1, d2, f]
        shapes = x.shape
        bs = shapes[0]
        # all
        out_all = x
        for layer in self.lin_all:
            out_all = layer(out_all)
        # rows
        pooled_rows = self._reduction(
            x, axis=1, keepdims=True
        )  # [b, d1, d2, f] -> [b, 1, d2, f]
        for layer in self.lin_n:
            pooled_rows = layer(pooled_rows) # [b, 1, d2, f] -> [b, 1, d2, f']
        out_rows = pooled_rows
        # cols
        pooled_cols = self._reduction(
            x, axis=2, keepdims=True
        )  # [b, d1, d2, f] -> [b, d1, 1, f]
        for layer in self.lin_m:
            pooled_cols = layer(pooled_cols) # [b, d1, 1, f] -> [b, d1, 1, f']
        out_cols = pooled_cols
        # both
        # todo: need to understand how we do this generic enough to move it into self._reduction.
        #  I think we can just flatten (1, 2) and call it on the flat axis
        # if self.reduction == "max":
        #     pooled_all, _ = torch.max(
        #         x.permute(0, 3, 1, 2).flatten(start_dim=2), dim=-1, keepdim=True
        #     )
        #     pooled_all = pooled_all.permute(0, 2, 1).unsqueeze(
        #         1
        #     )  # [b, d1, d2, f] -> [b, 1, 1, f]
        # else:
        # pooled_all = self._reduction(x, axis=(1, 2), keepdim=True)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        pooled_all = self._reduction(x, axis=2)
        pooled_all = jnp.expand_dims(jnp.expand_dims(pooled_all, 1), 1)  # [b, d1, d2, f] -> [b, 1, 1, f]

        for layer in self.lin_both:
            pooled_all = layer(pooled_all)  # [b, 1, 1, f] -> [b, 1, 1, f']
        out_both = pooled_all

        new_features = (
            out_all + out_rows + out_cols + out_both
        ) / 4.0  # [b, d1, d2, f']
        return new_features

class FromFirstLayer(BaseLayer):
    """Layer for mapping from first layer."""
    last_dim_is_output: bool = False
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        if self.last_dim_is_output:
            # i=0, j=L-1
            in_features = self.in_features * self.in_shape[0]  # d0 * in_features
            out_features = self.out_features * self.out_shape[1]  # dL * out_features
            self.layers = self._get_mlp(
                in_features=in_features, out_features=out_features, bias=self.bias
            )

        else:
            # i=0, j != L-1
            in_features = self.in_features * self.in_shape[0]  # d0 * in_features
            out_features = self.out_features  # out_features
            self.layers = self._get_mlp(
                in_features=in_features, out_features=out_features, bias=self.bias
            )

    def __call__(self, x):
        if self.last_dim_is_output:
            # i=0, j=L-1
            # (bs, d0, d1, in_features)
            # (bs, d0, in_features)
            x = self._reduction(x, axis=2)
            # (bs, dL * out_features)
            x = x.reshape(x.shape[0], -1)
            for layer in self.layers:
                x = layer(x)
            # (bs, d_{L-1}, dL, out_features)
            x = x.reshape(x.shape[0], self.out_shape[-1], self.out_features)
            x = jnp.expand_dims(x, 1)
            x = jnp.repeat(x, self.out_shape[0], axis=1)
        else:
            # i=0, j != L-1
            # (bs, d0, d1, in_features)
            # (bs, d0, in_features)
            x = self._reduction(x, axis=2)
            # (bs, out_features)
            x = x.reshape(x.shape[0], -1)
            for layer in self.layers:
                x = layer(x)
            # (bs, d_j, d_{j+1}, out_features)
            x = jnp.expand_dims(x, 1)
            x = jnp.expand_dims(x, 1)
            x = jnp.broadcast_to(x, (x.shape[0], *self.out_shape, x.shape[-1]))
        return x

class ToFirstLayer(BaseLayer):
    """Layer for mapping to first layer."""
    first_dim_is_output: bool = False
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        if self.first_dim_is_output:
            # i=L-1, j=0
            in_features = self.in_features * self.in_shape[-1]  # dL * in_features
            out_features = self.out_features * self.out_shape[0]  # d0 * out_features
            self.layers = self._get_mlp(in_features, out_features, bias=self.bias)

        else:
            # i!=L-1, j=0
            in_features = self.in_features  # in_features
            out_features = self.out_features * self.out_shape[0]  # d0 * out_features
            self.layers = self._get_mlp(in_features, out_features, bias=self.bias)

    def __call__(self, x):
        if self.first_dim_is_output:
            # i=L-1, j=0
            # (bs, d{L-1}, dL, in_features)
            # (bs, dL, in_features)
            x = self._reduction(x, axis=1)
            # (bs, d0 * out_features)
            x = x.reshape(x.shape[0], -1)
            for layer in self.layers:
                x = layer(x)
            # (bs, d0, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
            # (bs, d0, d1, out_features)
            x = jnp.expand_dims(x, 2)
            x = jnp.repeat(x, self.out_shape[-1], axis=2)
        else:
            # (bs, dj, d{j+1}, in_features)
            # (bs, in_features, dj * d{j+1})
            x = jnp.transpose(x, (0, 3, 1, 2))
            x = x.reshape(x.shape[0], x.shape[1], -1)
            # (bs, in_features)
            x = self._reduction(x, axis=2)
            # (bs, d0 * out_features)
            x = x.reshape(x.shape[0], -1)
            for layer in self.layers:
                x = layer(x)
            # (bs, d0, out_features)
            x = x.reshape(x.shape[0], self.out_shape[0], self.out_features)
            # (bs, d0, d1, out_features)
            x = jnp.expand_dims(x, 2)
            x = jnp.repeat(x, self.out_shape[-1], axis=2)
        return x

class FromLastLayer(BaseLayer):
    """Layer for mapping from last layer."""
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        self.layers = self._get_mlp(
            in_features=self.in_features * self.in_shape[-1],  # dL * in_features
            out_features=self.out_features,  # out_features
            bias=self.bias,
        )

    def __call__(self, x):
        # (bs, d{L-1}, dL, in_features)
        # (bs, dL, in_features)
        x = self._reduction(x, axis=1)
        # (bs, out_features)
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        # (bs, *out_shape, out_features)
        x = jnp.expand_dims(x, 1)
        x = jnp.expand_dims(x, 1)
        x = jnp.broadcast_to(x, (x.shape[0], *self.out_shape, x.shape[-1]))
        return x

class ToLastLayer(BaseLayer):
    """Layer for mapping to last layer."""
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        self.layers = self._get_mlp(
            in_features=self.in_features,  # dL * in_features
            out_features=self.out_features * self.out_shape[-1],  # out_features * dL
            bias=self.bias,
        )

    def __call__(self, x):
        # (bs, di, d{i+1}, in_features)
        # (bs, in_features, di * d{i+1})
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # (bs, in_features)
        x = self._reduction(x, axis=2)
        # (bs, dL * out_features)
        for layer in self.layers:
            x = layer(x)
        # (bs, dL, out_features)
        x = x.reshape(x.shape[0], self.out_shape[-1], self.out_features)
        # (bs, d{L-1}, dL, out_features)
        x = jnp.expand_dims(x, 1)
        x = jnp.repeat(x, self.out_shape[0], axis=1)
        return x

class NonNeighborInternalLayer(BaseLayer):
    """Layer for mapping between non-neighbor internal layers."""
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1

    def setup(self):
        super().setup()
        self.layers = self._get_mlp(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
        )

    def __call__(self, x):
        # (bs, di, d{i+1}, in_features)
        # (bs, in_features, di * d{i+1})
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # (bs, in_features)
        x = self._reduction(x, axis=2)
        # (bs, out_features)
        for layer in self.layers:
            x = layer(x)
        # (bs, *out_shape, out_features)
        x = jnp.expand_dims(x, 1)
        x = jnp.expand_dims(x, 1)
        x = jnp.broadcast_to(x, (x.shape[0], *self.out_shape, x.shape[-1]))
        return x

class WeightToWeightBlock(BaseLayer):
    """Block of layers mapping weights to weights."""
    shapes: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    diagonal: bool = False
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"

    def setup(self):
        super().setup()
        assert all([len(shape) == 2 for shape in self.shapes])
        assert len(self.shapes) > 2

        self.n_layers = len(self.shapes)

        layers = {}
        # construct layers:
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and abs(i - j) > 1:
                    continue
                if i == j:
                    if i == 0:
                        # W0 -> W0
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            first_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    elif j == self.n_layers - 1:
                        # W{L-1} -> W{L-1}
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            last_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    else:
                        layers[f"{i}_{j}"] = SetKroneckerSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            reduction=self.reduction,
                            bias=self.bias,
                            n_fc_layers=self.n_fc_layers,
                        )

                elif i == j - 1:
                    if i == 0:
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            first_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    elif j == self.n_layers - 1:
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            last_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    else:
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            in_index=i,
                            out_index=j,
                        )
                elif i == j + 1:
                    if j == 0:
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            last_dim_is_input=True,
                            in_index=i,
                            out_index=j,
                        )
                    elif i == self.n_layers - 1:
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            first_dim_is_output=True,
                            in_index=i,
                            out_index=j,
                        )
                    else:
                        layers[f"{i}_{j}"] = GeneralMatrixSetLayer(
                            in_features=self.in_features,
                            out_features=self.out_features,
                            in_shape=self.shapes[i],
                            out_shape=self.shapes[j],
                            reduction=self.reduction,
                            bias=self.bias,
                            num_heads=self.num_heads,
                            set_layer=self.set_layer,
                            n_fc_layers=self.n_fc_layers,
                            in_index=i,
                            out_index=j,
                        )
                elif i == 0:
                    layers[f"{i}_{j}"] = FromFirstLayer(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.shapes[i],
                        out_shape=self.shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        n_fc_layers=self.n_fc_layers,
                        last_dim_is_output=(
                            j == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                elif j == 0:
                    layers[f"{i}_{j}"] = ToFirstLayer(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.shapes[i],
                        out_shape=self.shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        n_fc_layers=self.n_fc_layers,
                        first_dim_is_output=(
                            i == self.n_layers - 1
                        ),  # todo: make sure this condition is correct
                    )
                elif i == self.n_layers - 1:
                    # j != i-1, 0
                    layers[f"{i}_{j}"] = FromLastLayer(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.shapes[i],
                        out_shape=self.shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        n_fc_layers=self.n_fc_layers,
                    )
                elif j == self.n_layers - 1:
                    layers[f"{i}_{j}"] = ToLastLayer(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.shapes[i],
                        out_shape=self.shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        n_fc_layers=self.n_fc_layers,
                    )
                else:
                    assert abs(i - j) > 1
                    layers[f"{i}_{j}"] = NonNeighborInternalLayer(
                        in_features=self.in_features,
                        out_features=self.out_features,
                        in_shape=self.shapes[i],
                        out_shape=self.shapes[j],
                        reduction=self.reduction,
                        bias=self.bias,
                        n_fc_layers=self.n_fc_layers,
                    )

        self.layers = layers

    def __call__(self, x: Tuple[jnp.ndarray, ...]):
        out_weights = [
            0.0,
        ] * len(x)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if self.diagonal and abs(i - j) > 1:
                    continue
                out_weights[j] = out_weights[j] + self.layers[f"{i}_{j}"](x[i])
        return tuple(out_weights)
    

if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)

    d0, d1, d2, d3, d4, d5 = 2, 10, 20, 30, 40, 1
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
    )
    print(len(matrices))
    weight_block = WeightToWeightBlock(
        in_features=12, out_features=24, shapes=tuple(m.shape[1:3] for m in matrices)
    )
    params = weight_block.init(key, matrices)
    out = weight_block.apply(params, matrices)
    print([o.shape for o in out])

    # perm test
    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)
    out_perm = weight_block.apply(params,
        (
            matrices[0][:, :, perm1, :],
            matrices[1][:, perm1, :, :][:, :, perm2, :],
            matrices[2][:, perm2, :, :][:, :, perm3, :],
            matrices[3][:, perm3, :, :][:, :, perm4, :],
            matrices[4][:, perm4, :, :],
        )
    )

    np.testing.assert_allclose(out[0][:, :, perm1, :], out_perm[0], atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        out[1][:, perm1, :, :][:, :, perm2, :], out_perm[1], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        out[2][:, perm2, :, :][:, :, perm3, :], out_perm[2], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        out[3][:, perm3, :, :][:, :, perm4, :], out_perm[3], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(out[4][:, perm4, :, :], out_perm[4], atol=1e-5, rtol=0)