import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from jax import random

from nn.layers_jax.weight_to_weight_jax import (
    FromFirstLayer,
    FromLastLayer,
    NonNeighborInternalLayer,
    ToFirstLayer,
    ToLastLayer,
    WeightToWeightBlock,
)
from nn.layers_jax.bias_to_bias_jax import BiasToBiasBlock
from nn.layers_jax.bias_to_weight_jax import BiasToWeightBlock
from nn.layers_jax.weight_to_bias_jax import WeightToBiasBlock
from nn.models_jax import DWSModel, DWSModelForClassification


def test_w_t_w_from_first():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = FromFirstLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[0],
        out_shape=shapes[-1],
        last_dim_is_output=True,
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = layer.init(key, matrices[0])

    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[0][:, :, perm1, :])
    out = layer.apply(params, matrices[0])
    np.testing.assert_allclose(out[:, perm4, :, :], out_perm, rtol=1e-5, atol=1e-5)


def test_w_t_w_to_first():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = ToFirstLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[-1],
        out_shape=shapes[0],
        first_dim_is_output=True,
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = layer.init(key, matrices[-1])

    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[-1][:, perm4, :, :])
    out = layer.apply(params, matrices[-1])
    np.testing.assert_allclose(out[:, :, perm1, :], out_perm, rtol=1e-5, atol=1e-5)


def test_w_t_w_from_last():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = FromLastLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[-1],
        out_shape=shapes[2],
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = layer.init(key, matrices[-1])

    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[-1][:, perm4, :, :])
    out = layer.apply(params, matrices[-1])
    np.testing.assert_allclose(
        out[:, perm2, :, :][:, :, perm3, :], out_perm, rtol=1e-5, atol=1e-5
    )


def test_w_t_w_to_last():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = ToLastLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[2],
        out_shape=shapes[-1],
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = layer.init(key, matrices[2])

    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[2][:, perm2, :, :][:, :, perm3, :])
    out = layer.apply(params, matrices[2])
    np.testing.assert_allclose(out[:, perm4, :, :], out_perm, rtol=1e-5, atol=1e-5)


def test_w_t_w_non_n():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = NonNeighborInternalLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[1],
        out_shape=shapes[3],
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = layer.init(key, matrices[1])

    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[1][:, perm1, :, :][:, :, perm2, :])
    out = layer.apply(params, matrices[1])
    np.testing.assert_allclose(
        out[:, perm2, :, :][:, :, perm3, :], out_perm, rtol=1e-5, atol=1e-5
    )


def test_weight_to_weight_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )

    weight_block = WeightToWeightBlock(
        in_features=12,
        out_features=24,
        shapes=tuple(m.shape[1:3] for m in matrices),
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = weight_block.init(key, matrices)

    out = weight_block.apply(params, matrices)

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = weight_block.apply(
        params,
        (
            matrices[0][:, :, perm1, :],
            matrices[1][:, perm1, :, :][:, :, perm2, :],
            matrices[2][:, perm2, :, :][:, :, perm3, :],
            matrices[3][:, perm3, :, :][:, :, perm4, :],
            matrices[4][:, perm4, :, :],
        ),
    )

    np.testing.assert_allclose(out[0][:, :, perm1, :], out_perm[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out[1][:, perm1, :, :][:, :, perm2, :], out_perm[1], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out[2][:, perm2, :, :][:, :, perm3, :], out_perm[2], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out[3][:, perm3, :, :][:, :, perm4, :], out_perm[3], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(out[4][:, perm4, :, :], out_perm[4], rtol=1e-5, atol=1e-5)


def test_bias_to_bias_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d1, 12)),
        random.normal(key, (4, d2, 12)),
        random.normal(key, (4, d3, 12)),
        random.normal(key, (4, d4, 12)),
        random.normal(key, (4, d5, 12)),
    )

    bias_block = BiasToBiasBlock(
        in_features=12,
        out_features=24,
        shapes=tuple(m.shape[1:2] for m in matrices),
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = bias_block.init(key, matrices)

    out = bias_block.apply(params, matrices)

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = bias_block.apply(
        params,
        (
            matrices[0][:, perm1, :],
            matrices[1][:, perm2, :],
            matrices[2][:, perm3, :],
            matrices[3][:, perm4, :],
            matrices[4],
        ),
    )

    np.testing.assert_allclose(out[0][:, perm1, :], out_perm[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1][:, perm2, :], out_perm[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[2][:, perm3, :], out_perm[2], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[3][:, perm4, :], out_perm[3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[4], out_perm[4], rtol=1e-5, atol=1e-5)


def test_bias_to_weight_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d1, 12)),
        random.normal(key, (4, d2, 12)),
        random.normal(key, (4, d3, 12)),
        random.normal(key, (4, d4, 12)),
        random.normal(key, (4, d5, 12)),
    )
    weights_shape = ((d0, d1), (d1, d2), (d2, d3), (d3, d4), (d4, d5))

    bias_block = BiasToWeightBlock(
        in_features=12,
        out_features=24,
        weight_shapes=weights_shape,
        bias_shapes=tuple(m.shape[1:2] for m in matrices),
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = bias_block.init(key, matrices)

    out = bias_block.apply(params, matrices)

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = bias_block.apply(
        params,
        (
            matrices[0][:, perm1, :],
            matrices[1][:, perm2, :],
            matrices[2][:, perm3, :],
            matrices[3][:, perm4, :],
            matrices[4],
        ),
    )

    np.testing.assert_allclose(out[0][:, :, perm1, :], out_perm[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out[1][:, perm1, :, :][:, :, perm2, :], out_perm[1], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out[2][:, perm2, :, :][:, :, perm3, :], out_perm[2], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out[3][:, perm3, :, :][:, :, perm4, :], out_perm[3], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(out[4][:, perm4, :, :], out_perm[4], rtol=1e-5, atol=1e-5)


def test_weight_to_bias_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    bias_shapes = ((d1,), (d2,), (d3,), (d4,), (d5,))

    weight_block = WeightToBiasBlock(
        in_features=12,
        out_features=24,
        weight_shapes=tuple(m.shape[1:3] for m in matrices),
        bias_shapes=bias_shapes,
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = weight_block.init(key, matrices)

    out = weight_block.apply(params, matrices)

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = weight_block.apply(
        params,
        (
            matrices[0][:, :, perm1, :],
            matrices[1][:, perm1, :, :][:, :, perm2, :],
            matrices[2][:, perm2, :, :][:, :, perm3, :],
            matrices[3][:, perm3, :, :][:, :, perm4, :],
            matrices[4][:, perm4, :, :],
        ),
    )

    np.testing.assert_allclose(out[0][:, perm1, :], out_perm[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1][:, perm2, :], out_perm[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[2][:, perm3, :], out_perm[2], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[3][:, perm4, :], out_perm[3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[4], out_perm[4], rtol=1e-5, atol=1e-5)


def test_model_invariance():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    bias = (
        random.normal(key, (4, d1, 12)),
        random.normal(key, (4, d2, 12)),
        random.normal(key, (4, d3, 12)),
        random.normal(key, (4, d4, 12)),
        random.normal(key, (4, d5, 12)),
    )

    model = DWSModel(
        weight_shapes=tuple(m.shape[1:3] for m in matrices),
        bias_shapes=tuple(b.shape[1:2] for b in bias),
        input_features=12,
        hidden_dim=64,
        n_hidden=2,
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = model.init(key, (matrices, bias))

    out = model.apply(params, (matrices, bias))

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = model.apply(
        params,
        (
            (
                matrices[0][:, :, perm1, :],
                matrices[1][:, perm1, :, :][:, :, perm2, :],
                matrices[2][:, perm2, :, :][:, :, perm3, :],
                matrices[3][:, perm3, :, :][:, :, perm4, :],
                matrices[4][:, perm4, :, :],
            ),
            (
                bias[0][:, perm1, :],
                bias[1][:, perm2, :],
                bias[2][:, perm3, :],
                bias[3][:, perm4, :],
                bias[4],
            ),
        ),
    )

    np.testing.assert_allclose(out[0][0][:, :, perm1, :], out_perm[0][0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out[0][1][:, perm1, :, :][:, :, perm2, :], out_perm[0][1], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out[0][2][:, perm2, :, :][:, :, perm3, :], out_perm[0][2], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out[0][3][:, perm3, :, :][:, :, perm4, :], out_perm[0][3], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(out[0][4][:, perm4, :, :], out_perm[0][4], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1][0][:, perm1, :], out_perm[1][0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1][1][:, perm2, :], out_perm[1][1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1][2][:, perm3, :], out_perm[1][2], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1][3][:, perm4, :], out_perm[1][3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[1][4], out_perm[1][4], rtol=1e-5, atol=1e-5)


def test_model_equivariance():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    bias = (
        random.normal(key, (4, d1, 12)),
        random.normal(key, (4, d2, 12)),
        random.normal(key, (4, d3, 12)),
        random.normal(key, (4, d4, 12)),
        random.normal(key, (4, d5, 12)),
    )

    model = DWSModelForClassification(
        weight_shapes=tuple(m.shape[1:3] for m in matrices),
        bias_shapes=tuple(b.shape[1:2] for b in bias),
        input_features=12,
        hidden_dim=64,
        n_hidden=2,
        n_classes=10,
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = model.init(key, (matrices, bias))

    out = model.apply(params, (matrices, bias))

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = model.apply(
        params,
        (
            (
                matrices[0][:, :, perm1, :],
                matrices[1][:, perm1, :, :][:, :, perm2, :],
                matrices[2][:, perm2, :, :][:, :, perm3, :],
                matrices[3][:, perm3, :, :][:, :, perm4, :],
                matrices[4][:, perm4, :, :],
            ),
            (
                bias[0][:, perm1, :],
                bias[1][:, perm2, :],
                bias[2][:, perm3, :],
                bias[3][:, perm4, :],
                bias[4],
            ),
        ),
    )

    np.testing.assert_allclose(out, out_perm, rtol=1e-5, atol=1e-5)


def test_model_equivariance_downsample():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    bias = (
        random.normal(key, (4, d1, 12)),
        random.normal(key, (4, d2, 12)),
        random.normal(key, (4, d3, 12)),
        random.normal(key, (4, d4, 12)),
        random.normal(key, (4, d5, 12)),
    )

    model = DWSModelForClassification(
        weight_shapes=tuple(m.shape[1:3] for m in matrices),
        bias_shapes=tuple(b.shape[1:2] for b in bias),
        input_features=12,
        hidden_dim=64,
        n_hidden=2,
        n_classes=10,
        input_dim_downsample=16,
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = model.init(key, (matrices, bias))

    out = model.apply(params, (matrices, bias))

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = model.apply(
        params,
        (
            (
                matrices[0][:, :, perm1, :],
                matrices[1][:, perm1, :, :][:, :, perm2, :],
                matrices[2][:, perm2, :, :][:, :, perm3, :],
                matrices[3][:, perm3, :, :][:, :, perm4, :],
                matrices[4][:, perm4, :, :],
            ),
            (
                bias[0][:, perm1, :],
                bias[1][:, perm2, :],
                bias[2][:, perm3, :],
                bias[3][:, perm4, :],
                bias[4],
            ),
        ),
    )

    np.testing.assert_allclose(out, out_perm, rtol=1e-5, atol=1e-5)


def test_model_equivariance_downsample_sab():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = random.PRNGKey(0)
    matrices = (
        random.normal(key, (4, d0, d1, 12)),
        random.normal(key, (4, d1, d2, 12)),
        random.normal(key, (4, d2, d3, 12)),
        random.normal(key, (4, d3, d4, 12)),
        random.normal(key, (4, d4, d5, 12)),
    )
    bias = (
        random.normal(key, (4, d1, 12)),
        random.normal(key, (4, d2, 12)),
        random.normal(key, (4, d3, 12)),
        random.normal(key, (4, d4, 12)),
        random.normal(key, (4, d5, 12)),
    )

    model = DWSModelForClassification(
        weight_shapes=tuple(m.shape[1:3] for m in matrices),
        bias_shapes=tuple(b.shape[1:2] for b in bias),
        input_features=12,
        hidden_dim=64,
        n_hidden=2,
        n_classes=10,
        input_dim_downsample=16,
        set_layer="sab",
    )

    # Initialize parameters
    key = random.PRNGKey(0)
    params = model.init(key, (matrices, bias))

    out = model.apply(params, (matrices, bias))

    # perm test
    perm1 = random.permutation(key, d1)
    perm2 = random.permutation(key, d2)
    perm3 = random.permutation(key, d3)
    perm4 = random.permutation(key, d4)
    out_perm = model.apply(
        params,
        (
            (
                matrices[0][:, :, perm1, :],
                matrices[1][:, perm1, :, :][:, :, perm2, :],
                matrices[2][:, perm2, :, :][:, :, perm3, :],
                matrices[3][:, perm3, :, :][:, :, perm4, :],
                matrices[4][:, perm4, :, :],
            ),
            (
                bias[0][:, perm1, :],
                bias[1][:, perm2, :],
                bias[2][:, perm3, :],
                bias[3][:, perm4, :],
                bias[4],
            ),
        ),
    )

    np.testing.assert_allclose(out, out_perm, rtol=1e-5, atol=1e-5) 