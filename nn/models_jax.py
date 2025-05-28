from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

from nn.layers_jax import (
    BN,
    DownSampleDWSLayer,
    Dropout,
    DWSLayer,
    InvariantLayer,
    ReLU,
)


class MLPModel(nn.Module):
    in_dim: int = 2208
    hidden_dim: int = 256
    n_hidden: int = 2
    bn: bool = False
    init_scale: float = 1.0

    def setup(self):
        # Create layers
        layers = []
        
        # First layer
        layers.append(nn.Dense(self.hidden_dim))
        layers.append(nn.relu)
        
        # Hidden layers
        for i in range(self.n_hidden):
            if i < self.n_hidden - 1:
                if not self.bn:
                    layers.append(nn.Dense(self.hidden_dim))
                    layers.append(nn.relu)
                else:
                    layers.append(nn.Dense(self.hidden_dim))
                    layers.append(nn.BatchNorm())
                    layers.append(nn.relu)
            else:
                layers.append(nn.Dense(self.in_dim))

        self.layers = layers

    def _init_model_params(self, params, scale):
        # Initialize parameters with Xavier initialization
        for layer in self.layers:
            if isinstance(layer, nn.Dense):
                out_c, in_c = layer.kernel.shape
                g = (2 * in_c / out_c) ** 0.5
                params = params.unfreeze()
                params[f"{layer.name}"]["kernel"] = params[f"{layer.name}"]["kernel"] * g * scale
                if "bias" in params[f"{layer.name}"]:
                    params[f"{layer.name}"]["bias"] = random.uniform(
                        random.PRNGKey(0),
                        params[f"{layer.name}"]["bias"].shape,
                        minval=-1e-4,
                        maxval=1e-4,
                    )
                params = params.freeze()
        return params

    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weight, bias = x
        bs = weight[0].shape[0]
        weight_shape, bias_shape = [w[0, :].shape for w in weight], [b[0, :].shape for b in bias]
        all_weights = weight + bias
        weight = jnp.concatenate([w.reshape(bs, -1) for w in all_weights], axis=-1)
        
        # Apply layers
        for layer in self.layers:
            weight = layer(weight)
            
        n_weights = sum([w[0].size for w in x[0]])
        weights = weight[:, :n_weights]
        biases = weight[:, n_weights:]
        
        weight, bias = [], []
        w_index = 0
        for s in weight_shape:
            weight.append(weights[:, w_index:w_index + jnp.prod(jnp.array(s))].reshape(bs, *s))
            w_index += jnp.prod(jnp.array(s))
            
        w_index = 0
        for s in bias_shape:
            bias.append(biases[:, w_index:w_index + jnp.prod(jnp.array(s))].reshape(bs, *s))
            w_index += jnp.prod(jnp.array(s))
            
        return tuple(weight), tuple(bias)


class MLPModelForClassification(nn.Module):
    in_dim: int
    hidden_dim: int = 256
    n_hidden: int = 2
    n_classes: int = 10
    bn: bool = False

    def setup(self):
        # Create layers
        self.layers = []
        
        # First layer
        self.layers.append(nn.Dense(self.hidden_dim))
        self.layers.append(nn.relu)
        
        # Hidden layers
        for _ in range(self.n_hidden):
            if not self.bn:
                self.layers.append(nn.Dense(self.hidden_dim))
                self.layers.append(nn.relu)
            else:
                self.layers.append(nn.Dense(self.hidden_dim))
                self.layers.append(nn.BatchNorm())
                self.layers.append(nn.relu)
                
        # Output layer
        self.layers.append(nn.Dense(self.n_classes))

    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        weight, bias = x
        all_weights = weight + bias
        weight = jnp.concatenate([w.reshape(w.shape[0], -1) for w in all_weights], axis=-1)
        
        # Apply layers
        for layer in self.layers:
            weight = layer(weight)
            
        return weight


class DWSModel(nn.Module):
    weight_shapes: Tuple[Tuple[int, int], ...]
    bias_shapes: Tuple[Tuple[int, ...], ...]
    input_features: int
    hidden_dim: int
    n_hidden: int = 2
    output_features: Optional[int] = None
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"
    input_dim_downsample: Optional[int] = None
    dropout_rate: float = 0.0
    add_skip: bool = False
    add_layer_skip: bool = False
    init_scale: float = 1e-4
    init_off_diag_scale_penalty: float = 1.0
    bn: bool = False
    diagonal: bool = False

    def setup(self):
        assert len(self.weight_shapes) > 2, "the current implementation only support input networks with M>2 layers."
        assert self.output_features is not None, "output_features must be specified. use hidden_dim if output_features is not specified."

        if self.add_skip:
            self.skip = nn.Dense(
                self.output_features,
                use_bias=self.bias,
                kernel_init=nn.initializers.ones,  # use for testing
                bias_init=nn.initializers.zeros,   # use for testing
                # kernel_init=lambda key, shape: jnp.ones(shape) / jnp.prod(jnp.array(shape)),
                # bias_init=lambda key, shape: jnp.zeros(shape),
            )

        if self.input_dim_downsample is None:
            layers = [
                DWSLayer(
                    weight_shapes=self.weight_shapes,
                    bias_shapes=self.bias_shapes,
                    in_features=self.input_features,
                    out_features=self.hidden_dim,
                    reduction=self.reduction,
                    bias=self.bias,
                    n_fc_layers=self.n_fc_layers,
                    num_heads=self.num_heads,
                    set_layer=self.set_layer,
                    add_skip=self.add_layer_skip,
                    init_scale=self.init_scale,
                    init_off_diag_scale_penalty=self.init_off_diag_scale_penalty,
                    diagonal=self.diagonal,
                ),
            ]
            
            for i in range(self.n_hidden):
                if self.bn:
                    layers.append(BN(self.hidden_dim, len(self.weight_shapes), len(self.bias_shapes)))

                layers.extend([
                    ReLU(),
                    Dropout(self.dropout_rate),
                    DWSLayer(
                        weight_shapes=self.weight_shapes,
                        bias_shapes=self.bias_shapes,
                        in_features=self.hidden_dim,
                        out_features=self.hidden_dim if i != (self.n_hidden - 1) else self.output_features,
                        reduction=self.reduction,
                        bias=self.bias,
                        n_fc_layers=self.n_fc_layers,
                        num_heads=self.num_heads if i != (self.n_hidden - 1) else 1,
                        set_layer=self.set_layer,
                        add_skip=self.add_layer_skip,
                        init_scale=self.init_scale,
                        init_off_diag_scale_penalty=self.init_off_diag_scale_penalty,
                        diagonal=self.diagonal,
                    ),
                ])
        else:
            new_weight_shapes = list(self.weight_shapes)
            new_weight_shapes[0] = (self.input_dim_downsample, self.weight_shapes[0][1])
            layers = [
                DownSampleDWSLayer(
                    original_weight_shapes=self.weight_shapes,
                    weight_shapes=tuple(new_weight_shapes),
                    bias_shapes=self.bias_shapes,
                    in_features=self.input_features,
                    out_features=self.hidden_dim,
                    reduction=self.reduction,
                    bias=self.bias,
                    n_fc_layers=self.n_fc_layers,
                    num_heads=self.num_heads,
                    set_layer=self.set_layer,
                    downsample_dim=self.input_dim_downsample,
                    add_skip=self.add_layer_skip,
                    init_scale=self.init_scale,
                    init_off_diag_scale_penalty=self.init_off_diag_scale_penalty,
                    diagonal=self.diagonal,
                ),
            ]            
            for i in range(self.n_hidden):
                if self.bn:
                    layers.append(BN(self.hidden_dim, len(self.weight_shapes), len(self.bias_shapes)))
                layers.extend([
                    ReLU(),
                    Dropout(self.dropout_rate),
                    DownSampleDWSLayer(
                        original_weight_shapes=self.weight_shapes,
                        weight_shapes=tuple(new_weight_shapes),
                        bias_shapes=self.bias_shapes,
                        in_features=self.hidden_dim,
                        out_features=self.hidden_dim if i != (self.n_hidden - 1) else self.output_features,
                        reduction=self.reduction,
                        bias=self.bias,
                        n_fc_layers=self.n_fc_layers,
                        num_heads=self.num_heads if i != (self.n_hidden - 1) else 1,
                        set_layer=self.set_layer,
                        downsample_dim=self.input_dim_downsample,
                        add_skip=self.add_layer_skip,
                        init_scale=self.init_scale,
                        init_off_diag_scale_penalty=self.init_off_diag_scale_penalty,
                        diagonal=self.diagonal,
                    ),
                ])
        self.layers = layers

    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]]):
        out = x
        for layer in self.layers:
            out = layer(out)
            
        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(self.skip(b) for b in x[1])
            weight_out = tuple(ws + w for w, ws in zip(out[0], skip_out[0]))
            bias_out = tuple(bs + b for b, bs in zip(out[1], skip_out[1]))
            out = weight_out, bias_out
            
        return out


class DWSModelForClassification(nn.Module):
    weight_shapes: Tuple[Tuple[int, int], ...]
    bias_shapes: Tuple[Tuple[int, ...], ...]
    input_features: int
    hidden_dim: int
    n_hidden: int = 2
    n_classes: int = 10
    reduction: str = "max"
    bias: bool = True
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "sab"
    n_out_fc: int = 1
    dropout_rate: float = 0.0
    input_dim_downsample: Optional[int] = None
    init_scale: float = 1.0
    init_off_diag_scale_penalty: float = 1.0
    bn: bool = False
    add_skip: bool = False
    add_layer_skip: bool = False
    equiv_out_features: Optional[int] = None
    diagonal: bool = False

    def setup(self):
        self.dws_model = DWSModel(
            weight_shapes=self.weight_shapes,
            bias_shapes=self.bias_shapes,
            input_features=self.input_features,
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
            output_features=self.hidden_dim if self.equiv_out_features is None else self.equiv_out_features,
            reduction=self.reduction,
            bias=self.bias,
            n_fc_layers=self.n_fc_layers,
            num_heads=self.num_heads,
            set_layer=self.set_layer,
            input_dim_downsample=self.input_dim_downsample,
            dropout_rate=self.dropout_rate,
            add_skip=self.add_skip,
            add_layer_skip=self.add_layer_skip,
            init_scale=self.init_scale,
            init_off_diag_scale_penalty=self.init_off_diag_scale_penalty,
            bn=self.bn,
            diagonal=self.diagonal,
        )
        
        self.classifier = nn.Dense(
            self.n_classes,
            kernel_init=nn.initializers.ones,  # use for testing
            bias_init=nn.initializers.zeros,   # use for testing
        )

    def __call__(self, x: Tuple[Tuple[jnp.ndarray], Tuple[jnp.ndarray]], return_equiv=False):
        equiv_out = self.dws_model(x)
        weight, bias = equiv_out
        all_weights = weight + bias
        weight = jnp.concatenate([w.reshape(w.shape[0], -1) for w in all_weights], axis=-1)
        out = self.classifier(weight)
        
        if return_equiv:
            return out, equiv_out
        return out 


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    weights = (
        jax.random.normal(key, (4, 784, 128, 1)),
        jax.random.normal(key, (4, 128, 128, 1)),
        jax.random.normal(key, (4, 128, 10, 1)),
    )
    biases = (
        jax.random.normal(key, (4, 128, 1)),
        jax.random.normal(key, (4, 128, 1)),
        jax.random.normal(key, (4, 10, 1)),
    )
    in_dim = sum([w[0, :].size for w in weights]) + sum(
        [w[0, :].size for w in biases]
    )
    weight_shapes = tuple(w.shape[1:3] for w in weights)
    bias_shapes = tuple(b.shape[1:2] for b in biases)

    model = MLPModel(in_dim=in_dim, hidden_dim=128, n_hidden=2)
    params = model.init(key, (weights, biases))
    out = model.apply(params, (weights, biases))

    for weight, bias in zip(out[0], out[1]):
        for w, b in zip(weight, bias):
            print(w.shape, b.shape)