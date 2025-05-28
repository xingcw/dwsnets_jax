import jax
import numpy as np
import torch
import pytest
import random

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

torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def test_w_t_w_from_first():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
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
    key = jax.random.PRNGKey(0)
    params = layer.init(key, matrices[0])

    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[0][:, :, perm1, :])
    out = layer.apply(params, matrices[0])
    np.testing.assert_allclose(out[:, perm4, :, :], out_perm, rtol=1e-5, atol=1e-5)


def test_w_t_w_to_first():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
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
    key = jax.random.PRNGKey(0)
    params = layer.init(key, matrices[-1])

    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[-1][:, perm4, :, :])
    out = layer.apply(params, matrices[-1])
    np.testing.assert_allclose(out[:, :, perm1, :], out_perm, rtol=1e-5, atol=1e-5)


def test_w_t_w_from_last():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = FromLastLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[-1],
        out_shape=shapes[2],
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = layer.init(key, matrices[-1])

    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[-1][:, perm4, :, :])
    out = layer.apply(params, matrices[-1])
    np.testing.assert_allclose(
        out[:, perm2, :, :][:, :, perm3, :], out_perm, rtol=1e-5, atol=1e-5
    )


def test_w_t_w_to_last():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = ToLastLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[2],
        out_shape=shapes[-1],
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = layer.init(key, matrices[2])

    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[2][:, perm2, :, :][:, :, perm3, :])
    out = layer.apply(params, matrices[2])
    np.testing.assert_allclose(out[:, perm4, :, :], out_perm, rtol=1e-5, atol=1e-5)


def test_w_t_w_non_n():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
    )
    shapes = tuple(m.shape[1:3] for m in matrices)

    layer = NonNeighborInternalLayer(
        in_features=12,
        out_features=24,
        in_shape=shapes[1],
        out_shape=shapes[3],
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = layer.init(key, matrices[1])

    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)

    out_perm = layer.apply(params, matrices[1][:, perm1, :, :][:, :, perm2, :])
    out = layer.apply(params, matrices[1])
    np.testing.assert_allclose(
        out[:, perm2, :, :][:, :, perm3, :], out_perm, rtol=1e-5, atol=1e-5
    )


def test_weight_to_weight_block():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
    )

    weight_block = WeightToWeightBlock(
        in_features=12,
        out_features=24,
        shapes=tuple(m.shape[1:3] for m in matrices),
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = weight_block.init(key, matrices)

    out = weight_block.apply(params, matrices)

    # perm test
    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)
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
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d1, 12)),
        jax.random.normal(key, (4, d2, 12)),
        jax.random.normal(key, (4, d3, 12)),
        jax.random.normal(key, (4, d4, 12)),
        jax.random.normal(key, (4, d5, 12)),
    )

    bias_block = BiasToBiasBlock(
        in_features=12,
        out_features=24,
        shapes=tuple(m.shape[1:2] for m in matrices),
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = bias_block.init(key, matrices)

    out = bias_block.apply(params, matrices)

    # perm test
    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)
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
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d1, 12)),
        jax.random.normal(key, (4, d2, 12)),
        jax.random.normal(key, (4, d3, 12)),
        jax.random.normal(key, (4, d4, 12)),
        jax.random.normal(key, (4, d5, 12)),
    )
    weights_shape = ((d0, d1), (d1, d2), (d2, d3), (d3, d4), (d4, d5))

    bias_block = BiasToWeightBlock(
        in_features=12,
        out_features=24,
        weight_shapes=weights_shape,
        bias_shapes=tuple(m.shape[1:2] for m in matrices),
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = bias_block.init(key, matrices)

    out = bias_block.apply(params, matrices)

    # perm test
    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)
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
    key = jax.random.PRNGKey(0)
    matrices = (
        jax.random.normal(key, (4, d0, d1, 12)),
        jax.random.normal(key, (4, d1, d2, 12)),
        jax.random.normal(key, (4, d2, d3, 12)),
        jax.random.normal(key, (4, d3, d4, 12)),
        jax.random.normal(key, (4, d4, d5, 12)),
    )
    bias_shapes = ((d1,), (d2,), (d3,), (d4,), (d5,))

    weight_block = WeightToBiasBlock(
        in_features=12,
        out_features=24,
        weight_shapes=tuple(m.shape[1:3] for m in matrices),
        bias_shapes=bias_shapes,
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = weight_block.init(key, matrices)

    out = weight_block.apply(params, matrices)

    # perm test
    perm1 = jax.random.permutation(key, d1)
    perm2 = jax.random.permutation(key, d2)
    perm3 = jax.random.permutation(key, d3)
    perm4 = jax.random.permutation(key, d4)
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
    weights = (
        np.random.randn(4, d0, d1, 2).astype(np.float64),
        np.random.randn(4, d1, d2, 2).astype(np.float64),
        np.random.randn(4, d2, d3, 2).astype(np.float64),
        np.random.randn(4, d3, d4, 2).astype(np.float64),
        np.random.randn(4, d4, d5, 2).astype(np.float64),
    )
    biases = (
        np.random.randn(4, d1, 2).astype(np.float64),
        np.random.randn(4, d2, 2).astype(np.float64),
        np.random.randn(4, d3, 2).astype(np.float64),
        np.random.randn(4, d4, 2).astype(np.float64),
        np.random.randn(4, d5, 2).astype(np.float64),
    )

    model = DWSModelForClassification(
        input_features=2,
        n_classes=10,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=16,
        dropout_rate=0.0,
    )
    key = jax.random.PRNGKey(0)
    params = model.init(key, (weights, biases))
    out = model.apply(params, (weights, biases))
    # perm test
    perm1 = np.random.permutation(d1)
    perm2 = np.random.permutation(d2)
    perm3 = np.random.permutation(d3)
    perm4 = np.random.permutation(d4)
    out_perm = model.apply(
        params,
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :][:, :, perm4, :],
                weights[4][:, perm4, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3][:, perm4, :],
                biases[4],
            ),
        )
    )

    np.testing.assert_allclose(out, out_perm, atol=1e-5, rtol=1e-5)


def test_model_equivariance():
    d0, d1, d2, d3, d4, d5 = 2, 32, 32, 32, 32, 3
    weights = (
        np.random.randn(4, d0, d1, 2).astype(np.float64),
        np.random.randn(4, d1, d2, 2).astype(np.float64),
        np.random.randn(4, d2, d3, 2).astype(np.float64),
        np.random.randn(4, d3, d4, 2).astype(np.float64),
        np.random.randn(4, d4, d5, 2).astype(np.float64),
    )
    biases = (
        np.random.randn(4, d1, 2).astype(np.float64),
        np.random.randn(4, d2, 2).astype(np.float64),
        np.random.randn(4, d3, 2).astype(np.float64),
        np.random.randn(4, d4, 2).astype(np.float64),
        np.random.randn(4, d5, 2).astype(np.float64),
    )

    model = DWSModel(
        input_features=2,
        output_features=8,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=16,
        dropout_rate=0.0,
        bias=True,
    )
    key = jax.random.PRNGKey(0)
    params = model.init(key, (weights, biases))
    out = model.apply(params, (weights, biases))
    # perm test
    perm1 = np.random.permutation(d1)
    perm2 = np.random.permutation(d2)
    perm3 = np.random.permutation(d3)
    perm4 = np.random.permutation(d4)
    out_perm = model.apply(
        params,
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :][:, :, perm4, :],
                weights[4][:, perm4, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3][:, perm4, :],
                biases[4],
            ),
        )
    )

    out_weights = out[0]
    out_weights_perm = out_perm[0]
    np.testing.assert_allclose(
        out_weights[0][:, :, perm1, :], out_weights_perm[0], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_weights[1][:, perm1, :, :][:, :, perm2, :],
        out_weights_perm[1],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[2][:, perm2, :, :][:, :, perm3, :],
        out_weights_perm[2],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[3][:, perm3, :, :][:, :, perm4, :],
        out_weights_perm[3],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[4][:, perm4, :, :], out_weights_perm[4], atol=1e-4, rtol=1e-5
    )

    out_biases = out[1]
    out_biases_perm = out_perm[1]
    np.testing.assert_allclose(
        out_biases[0][:, perm1, :], out_biases_perm[0], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[1][:, perm2, :], out_biases_perm[1], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[2][:, perm3, :], out_biases_perm[2], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[3][:, perm4, :], out_biases_perm[3], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(out_biases[4], out_biases_perm[4], atol=1e-4, rtol=1e-5)


def test_model_equivariance_downsample():
    d0, d1, d2, d3, d4, d5 = 64, 32, 32, 32, 32, 3
    weights = (
        np.random.randn(4, d0, d1, 2),
        np.random.randn(4, d1, d2, 2),
        np.random.randn(4, d2, d3, 2),
        np.random.randn(4, d3, d4, 2),
        np.random.randn(4, d4, d5, 2),
    )
    biases = (
        np.random.randn(4, d1, 2),
        np.random.randn(4, d2, 2),
        np.random.randn(4, d3, 2),
        np.random.randn(4, d4, 2),
        np.random.randn(4, d5, 2),
    )

    model = DWSModel(
        input_features=2,
        output_features=8,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=16,
        input_dim_downsample=16,
        dropout_rate=0.0,
    )
    key = jax.random.PRNGKey(0)
    params = model.init(key, (weights, biases))
    out = model.apply(params, (weights, biases))
    # perm test
    perm1 = np.random.permutation(d1)
    perm2 = np.random.permutation(d2)
    perm3 = np.random.permutation(d3)
    perm4 = np.random.permutation(d4)
    out_perm = model.apply(
        params,
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :][:, :, perm4, :],
                weights[4][:, perm4, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3][:, perm4, :],
                biases[4],
            ),
        )
    )

    out_weights = out[0]
    out_weights_perm = out_perm[0]
    np.testing.assert_allclose(
        out_weights[0][:, :, perm1, :], out_weights_perm[0], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_weights[1][:, perm1, :, :][:, :, perm2, :],
        out_weights_perm[1],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[2][:, perm2, :, :][:, :, perm3, :],
        out_weights_perm[2],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[3][:, perm3, :, :][:, :, perm4, :],
        out_weights_perm[3],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[4][:, perm4, :, :], out_weights_perm[4], atol=1e-4, rtol=1e-5
    )

    out_biases = out[1]
    out_biases_perm = out_perm[1]
    np.testing.assert_allclose(
        out_biases[0][:, perm1, :], out_biases_perm[0], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[1][:, perm2, :], out_biases_perm[1], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[2][:, perm3, :], out_biases_perm[2], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[3][:, perm4, :], out_biases_perm[3], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(out_biases[4], out_biases_perm[4], atol=1e-4, rtol=1e-5)


def test_model_equivariance_downsample_sab():
    d0, d1, d2, d3, d4 = 28 * 28, 128, 128, 128, 10
    weights = (
        np.random.randn(4, d0, d1, 1).astype(np.float64),
        np.random.randn(4, d1, d2, 1).astype(np.float64),
        np.random.randn(4, d2, d3, 1).astype(np.float64),
        np.random.randn(4, d3, d4, 1).astype(np.float64),
    )
    biases = (
        np.random.randn(4, d1, 1).astype(np.float64),
        np.random.randn(4, d2, 1).astype(np.float64),
        np.random.randn(4, d3, 1).astype(np.float64),
        np.random.randn(4, d4, 1).astype(np.float64),
    )

    model = DWSModel(
        input_features=1,
        output_features=1,
        weight_shapes=tuple(m.shape[1:3] for m in weights),
        bias_shapes=tuple(m.shape[1:2] for m in biases),
        hidden_dim=64,
        input_dim_downsample=16,
        dropout_rate=0.0,
        add_skip=True,
    )
    key = jax.random.PRNGKey(0)
    params = model.init(key, (weights, biases))
    out = model.apply(params, (weights, biases))
    # perm test
    perm1 = np.random.permutation(d1)
    perm2 = np.random.permutation(d2)
    perm3 = np.random.permutation(d3)
    perm4 = np.random.permutation(d4)
    out_perm = model.apply(
        params,
        (
            (
                weights[0][:, :, perm1, :],
                weights[1][:, perm1, :, :][:, :, perm2, :],
                weights[2][:, perm2, :, :][:, :, perm3, :],
                weights[3][:, perm3, :, :],
            ),
            (
                biases[0][:, perm1, :],
                biases[1][:, perm2, :],
                biases[2][:, perm3, :],
                biases[3],
            ),
        )
    )

    out_weights = out[0]
    out_weights_perm = out_perm[0]
    np.testing.assert_allclose(
        out_weights[0][:, :, perm1, :], out_weights_perm[0], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_weights[1][:, perm1, :, :][:, :, perm2, :],
        out_weights_perm[1],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[2][:, perm2, :, :][:, :, perm3, :],
        out_weights_perm[2],
        atol=1e-4,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out_weights[3][:, perm3, :, :], out_weights_perm[3], atol=1e-4, rtol=1e-5
    )

    out_biases = out[1]
    out_biases_perm = out_perm[1]
    np.testing.assert_allclose(
        out_biases[0][:, perm1, :], out_biases_perm[0], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[1][:, perm2, :], out_biases_perm[1], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(
        out_biases[2][:, perm3, :], out_biases_perm[2], atol=1e-4, rtol=1e-5
    )
    np.testing.assert_allclose(out_biases[3], out_biases_perm[3], atol=1e-4, rtol=1e-5)


if __name__ == "__main__":

    test_model_equivariance_downsample()