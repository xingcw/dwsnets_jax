import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple


class BaseLayer(nn.Module):
    in_features: int
    out_features: int
    in_shape: Optional[Tuple] = None
    out_shape: Optional[Tuple] = None
    bias: bool = True
    reduction: str = "mean"
    n_fc_layers: int = 1
    num_heads: int = 8
    set_layer: str = "ds"

    def setup(self):
        assert self.set_layer in ["ds", "sab"]
        assert self.reduction in ["mean", "sum", "attn", "max"]
        self.b = None

    def _get_mlp(self, in_features, out_features, bias=False):
        layers = []
        layers.append(nn.Dense(
            out_features,
            use_bias=bias,
            kernel_init=nn.initializers.ones, # for testing
            bias_init=nn.initializers.zeros,
        ))
        for _ in range(self.n_fc_layers - 1):
            layers.append(nn.relu)
            layers.append(nn.Dense(
                out_features,
                use_bias=bias,
                kernel_init=nn.initializers.ones, # for testing
                bias_init=nn.initializers.zeros,
            ))
        return layers

    def _reduction(self, x, axis=1, keepdims=False):
        if self.reduction == "mean":
            x = jnp.mean(x, axis=axis, keepdims=keepdims)
        elif self.reduction == "sum":
            x = jnp.sum(x, axis=axis, keepdims=keepdims)
        elif self.reduction == "attn":
            assert x.ndim == 3
            raise NotImplementedError
        elif self.reduction == "max":
            x = jnp.max(x, axis=axis, keepdims=keepdims)
        else:
            raise ValueError(f"invalid reduction, got {self.reduction}")
        return x

class MAB(nn.Module):
    dim_Q: int
    dim_K: int
    dim_V: int
    num_heads: int
    ln: bool = False

    def setup(self):
        self.fc_q = nn.Dense(
            self.dim_V,
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.fc_k = nn.Dense(
            self.dim_V,
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.fc_v = nn.Dense(
            self.dim_V,
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        if self.ln:
            self.ln0 = nn.LayerNorm()
            self.ln1 = nn.LayerNorm()
        self.fc_o = nn.Dense(
            self.dim_V,
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        num_splits = Q.shape[2] // dim_split
        Q_ = jnp.concatenate(jnp.split(Q, num_splits, axis=-1), axis=0)
        K_ = jnp.concatenate(jnp.split(K, num_splits, axis=-1), axis=0)
        V_ = jnp.concatenate(jnp.split(V, num_splits, axis=-1), axis=0)

        A = jax.nn.softmax(Q_ @ K_.transpose(0, 2, 1) / jnp.sqrt(self.dim_V), axis=2)

        O = Q_ + jnp.einsum('bqk,bkd->bqd', A, V_)
        num_splits = O.shape[0] // Q.shape[0]
        O = jnp.concatenate(jnp.split(O, num_splits, axis=0), axis=-1)
        
        if self.ln:
            O = self.ln0(O)
        O = O + nn.relu(self.fc_o(O))
        if self.ln:
            O = self.ln1(O)
        return O

class SAB(BaseLayer):
    def setup(self):
        super().setup()
        self.mab = MAB(
            dim_Q=self.in_features,
            dim_K=self.in_features,
            dim_V=self.out_features,
            num_heads=self.num_heads
        )

    def __call__(self, X):
        return self.mab(X, X)

class SetLayer(BaseLayer):
    def setup(self):
        super().setup()
        self.Gamma = self._get_mlp(self.in_features, self.out_features, bias=self.bias)
        self.Lambda = self._get_mlp(self.in_features, self.out_features, bias=False)
        if self.reduction == "attn":
            self.attn = Attn(dim=self.in_features)

    def __call__(self, x):
        # set dim is 1
        if self.reduction == "mean":
            xm = jnp.mean(x, axis=1, keepdims=True)
        elif self.reduction == "sum":
            xm = jnp.sum(x, axis=1, keepdims=True)
        elif self.reduction == "attn":
            xm = self.attn(jnp.swapaxes(x, -1, -2), keepdims=True)
            xm = jnp.swapaxes(xm, -1, -2)
        else:
            xm = jnp.max(x, axis=1, keepdims=True)

        # Apply MLP layers
        for layer in self.Lambda:
            xm = layer(xm)
        for layer in self.Gamma:
            x = layer(x)
        
        x = x - xm
        return x

class GeneralSetLayer(BaseLayer):
    def setup(self):
        super().setup()
        self.set_layers = {
            "ds": SetLayer(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=self.bias,
                reduction=self.reduction,
                n_fc_layers=self.n_fc_layers,
            ),
            "sab": SAB(
                in_features=self.in_features,
                out_features=self.out_features,
                num_heads=self.num_heads
            ),
        }
        self.layers = self.set_layers[self.set_layer]

    def __call__(self, x):
        return self.layers(x)

class Attn(nn.Module):
    dim: int

    def setup(self):
        self.query = self.param('query', nn.initializers.ones, (self.dim,))

    def __call__(self, x, keepdims=False):
        # Note: reduction is applied to last dim. For example for (bs, d, d') we compute d' attn weights
        # by multiplying over d.
            
        attn = jnp.sum(jnp.swapaxes(x, -1, -2) * self.query, axis=-1)
        attn = jax.nn.softmax(attn, axis=-1)
        if x.ndim == 3:
            attn = jnp.expand_dims(attn, 1)
        elif x.ndim == 4:
            attn = jnp.expand_dims(attn, 2)
        else:
            raise ValueError(f"invalid input dimension, got {x.ndim}")
        output = jnp.sum(x * attn, axis=-1, keepdims=keepdims)
        return output