from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import field

from nn.layers_jax.base_jax import BaseLayer, GeneralSetLayer
from nn.layers_jax.bias_to_bias_jax import BiasToBiasBlock
from nn.layers_jax.bias_to_weight_jax import BiasToWeightBlock
from nn.layers_jax.weight_to_bias_jax import WeightToBiasBlock
from nn.layers_jax.weight_to_weight_jax import WeightToWeightBlock


class BN(nn.Module):
    num_features: int
    n_weights: int
    n_biases: int
    use_running_average: bool = True

    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]], training: bool = True):
        weights, biases = x
        new_weights, new_biases = [None] * len(weights), [None] * len(biases)
        
        for i, w in enumerate(weights):
            shapes = w.shape
            # Create BatchNorm layers for weights
            bn = nn.BatchNorm(
                use_running_average=not training,
                momentum=0.9,
                epsilon=1e-5,
                dtype=jnp.float32,
                name=f'weights_bn_{i}'
            )
            # Reshape and apply batch norm
            w_reshaped = w.transpose(0, 3, 1, 2).reshape(w.shape[0], -1, w.shape[1] * w.shape[2])
            w_normed = bn(w_reshaped)
            new_weights[i] = w_normed.reshape(shapes)

        for i, b in enumerate(biases):
            # Create BatchNorm layers for biases
            bn = nn.BatchNorm(
                use_running_average=not training,
                momentum=0.9,
                epsilon=1e-5,
                dtype=jnp.float32,
                name=f'biases_bn_{i}'
            )
            # Reshape and apply batch norm
            b_reshaped = b.transpose(0, 2, 1)
            b_normed = bn(b_reshaped)
            new_biases[i] = b_normed.transpose(0, 2, 1)

        return tuple(new_weights), tuple(new_biases)


class ReLU(nn.Module):
    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weights, biases = x
        return tuple(jax.nn.relu(t) for t in weights), tuple(jax.nn.relu(t) for t in biases)


class LeakyReLU(nn.Module):
    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weights, biases = x
        return tuple(jax.nn.leaky_relu(t) for t in weights), tuple(jax.nn.relu(t) for t in biases)


class Dropout(nn.Module):
    rate: float = 0.1
    deterministic: bool = False

    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]], training: bool = True):
        weights, biases = x
        dropout_fn = nn.Dropout(rate=self.rate, deterministic=not training)
        return (
            tuple(dropout_fn(t) for t in weights),
            tuple(dropout_fn(t) for t in biases)
        )


class DWSLayer(BaseLayer):
    weight_shapes: Tuple[Tuple[int, int], ...] = None
    bias_shapes: Tuple[Tuple[int,], ...] = None
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"
    add_skip: bool = False
    init_scale: float = 1.0
    init_off_diag_scale_penalty: float = 1.0
    diagonal: bool = False

    def setup(self):
        self.n_matrices = len(self.weight_shapes) + len(self.bias_shapes)

        self.weight_to_weight = WeightToWeightBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            shapes=self.weight_shapes,
            bias=self.bias,
            reduction=self.reduction,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer=self.set_layer,
            diagonal=self.diagonal,
        )

        self.bias_to_bias = BiasToBiasBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            shapes=self.bias_shapes,
            bias=self.bias,
            reduction=self.reduction,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer=self.set_layer,
            diagonal=self.diagonal,
        )

        self.bias_to_weight = BiasToWeightBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            bias_shapes=self.bias_shapes,
            weight_shapes=self.weight_shapes,
            bias=self.bias,
            reduction=self.reduction,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer=self.set_layer,
            diagonal=self.diagonal,
        )

        self.weight_to_bias = WeightToBiasBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            bias_shapes=self.bias_shapes,
            weight_shapes=self.weight_shapes,
            bias=self.bias,
            reduction=self.reduction,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer=self.set_layer,
            diagonal=self.diagonal,
        )

        if self.add_skip:
            self.skip = self._get_mlp(self.in_features, self.out_features, bias=self.bias)

    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weights, biases = x
        
        new_weights_from_weights = self.weight_to_weight(weights)
        new_weights_from_biases = self.bias_to_weight(biases)
        new_biases_from_biases = self.bias_to_bias(biases)
        new_biases_from_weights = self.weight_to_bias(weights)

        # Add and normalize by the number of matrices
        new_weights = tuple(
            (w0 + w1) / self.n_matrices
            for w0, w1 in zip(new_weights_from_weights, new_weights_from_biases)
        )
        new_biases = tuple(
            (b0 + b1) / self.n_matrices
            for b0, b1 in zip(new_biases_from_biases, new_biases_from_weights)
        )

        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(self.skip(b) for b in x[1])
            new_weights = tuple(ws + w for w, ws in zip(new_weights, skip_out[0]))
            new_biases = tuple(bs + b for b, bs in zip(new_biases, skip_out[1]))

        return new_weights, new_biases


class DownSampleDWSLayer(DWSLayer):
    downsample_dim: int = None
    original_weight_shapes: Tuple[Tuple[int, int], ...] = None

    def setup(self):
        d0 = self.original_weight_shapes[0][0]
        super().setup()
        
        self.down_sample = GeneralSetLayer(
            in_features=d0,
            out_features=self.downsample_dim,
            reduction="attn",
            bias=self.bias,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer="ds",
        )

        self.up_sample = GeneralSetLayer(
            in_features=self.downsample_dim,
            out_features=d0,
            reduction="attn",
            bias=self.bias,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer="ds",
        )

        self.skip_layers = self._get_mlp(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
        )

    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weights, biases = x

        # Down-sample
        w0 = weights[0]
        w0_skip = w0
        for layer in self.skip_layers:
            w0_skip = layer(w0_skip)
        
        bs, d0, d1, _ = w0.shape
        
        # Reshape and apply down-sampling
        w0 = jnp.transpose(w0, (0, 3, 2, 1))
        w0 = self.down_sample(w0)
        w0 = jnp.transpose(w0, (0, 3, 2, 1))
        
        weights = list(weights)
        weights[0] = w0

        # Apply DWS layer
        weights, biases = super().__call__((tuple(weights), biases))

        # Up-sample
        w0 = weights[0]
        w0 = jnp.transpose(w0, (0, 3, 2, 1))
        w0 = self.up_sample(w0)
        w0 = jnp.transpose(w0, (0, 3, 2, 1))
        
        weights = list(weights)
        weights[0] = w0 + w0_skip  # Add skip connection

        return tuple(weights), biases


class InvariantLayer(BaseLayer):
    weight_shapes: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    bias_shapes: Tuple[Tuple[int,], ...] = field(default_factory=tuple)
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1

    def setup(self):
        n_layers = len(self.weight_shapes) + len(self.bias_shapes)
        self.mlp_layers = self._get_mlp(
            in_features=(
                self.in_features * (n_layers - 3)
                +
                # in_features * d0 - first weight matrix
                self.in_features * self.weight_shapes[0][0]
                +
                # in_features * dL - last weight matrix
                self.in_features * self.weight_shapes[-1][-1]
                +
                # in_features * dL - last bias
                self.in_features * self.bias_shapes[-1][-1]
            ),
            out_features=self.out_features,
            bias=self.bias,
        )

    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weights, biases = x
        
        # Handle first and last matrices specially
        first_w, last_w = weights[0], weights[-1]
        pooled_first_w = first_w.transpose(0, 2, 1, 3)
        # (bs, d1, d0 * in_features)
        pooled_first_w = pooled_first_w.reshape(pooled_first_w.shape[0], pooled_first_w.shape[1], -1)
        # (bs, d{L-1}, dL * in_features)
        pooled_last_w = last_w.reshape(last_w.shape[0], last_w.shape[1], -1)
        # (bs, d0 * in_features)
        pooled_first_w = self._reduction(pooled_first_w, axis=1)
        # (bs, dL * in_features)
        pooled_last_w = self._reduction(pooled_last_w, axis=1)
        
        # Handle last bias specially
        last_b = biases[-1]
        # (bs, dL * in_features)
        pooled_last_b = last_b.reshape(last_b.shape[0], -1)

        # Concatenate intermediate weights
        pooled_weights = jnp.concatenate([
            self._reduction(w.transpose(0, 3, 1, 2).reshape(w.shape[0], -1, w.shape[1] * w.shape[2]), axis=2)
            for w in weights[1:-1]
        ], axis=-1)# (bs, (len(weights) - 2) * in_features)
        # (bs, (len(weights) - 2) * in_features + d0 * in_features + dL * in_features)
        pooled_weights = jnp.concatenate((pooled_weights, pooled_first_w, pooled_last_w), axis=-1)

        # Concatenate biases
        pooled_biases = jnp.concatenate(
            [self._reduction(b, axis=1) for b in biases[:-1]], axis=-1
        )# (bs, (len(biases) - 1) * in_features)
        # (bs, (len(biases) - 1) * in_features + dL * in_features)
        pooled_biases = jnp.concatenate((pooled_biases, pooled_last_b), axis=-1)

        # Combine all features
        pooled_all = jnp.concatenate([pooled_weights, pooled_biases], axis=-1)
        # (bs, (num layers - 3) * in_features + d0 * in_features + dL * in_features + dL * in_features)
        for layer in self.mlp_layers:
            pooled_all = layer(pooled_all)
        return pooled_all


class NaiveInvariantLayer(BaseLayer):
    weight_shapes: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    bias_shapes: Tuple[Tuple[int,], ...] = field(default_factory=tuple)
    bias: bool = True
    reduction: str = "max"
    n_fc_layers: int = 1

    def setup(self):
        n_layers = len(self.weight_shapes) + len(self.bias_shapes)
        self.layer = self._get_mlp(
            in_features=self.in_features * n_layers,
            out_features=self.out_features,
            bias=self.bias
        )

    @nn.compact
    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weights, biases = x
        
        pooled_weights = jnp.concatenate([
            self._reduction(w.transpose(0, 3, 1, 2).reshape(w.shape[0], -1, w.shape[1] * w.shape[2]), axis=2)
            for w in weights
        ], axis=-1)

        pooled_biases = jnp.concatenate(
            [self._reduction(b, axis=1) for b in biases], axis=-1
        )

        pooled_all = jnp.concatenate([pooled_weights, pooled_biases], axis=-1)
        return self.layer(pooled_all) 