import jax
import jax.numpy as jnp
import numpy as np
import torch
import random
import pytest
from nn.layers.bias_to_weight import (
    SameLayer as TorchSameLayer,
    SuccessiveLayers as TorchSuccessiveLayers,
    NonNeighborInternalLayer as TorchNonNeighborInternalLayer,
    BiasToWeightBlock as TorchBiasToWeightBlock,
)
from nn.layers_jax.bias_to_weight_jax import (
    SameLayer as JaxSameLayer,
    SuccessiveLayers as JaxSuccessiveLayers,
    NonNeighborInternalLayer as JaxNonNeighborInternalLayer,
    BiasToWeightBlock as JaxBiasToWeightBlock,
)

torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def test_same_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (10,)
    out_shape = (8, 10)
    
    X = np.random.randn(batch_size, in_shape[0], in_features)
    
    # Test normal case
    torch_model = TorchSameLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    jax_model = JaxSameLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)
    
    # Test input layer case
    torch_model = TorchSameLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        is_input_layer=True,
    )
    jax_model = JaxSameLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        is_input_layer=True,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)
    
    # Test output layer case
    torch_model = TorchSameLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        is_output_layer=True,
    )
    jax_model = JaxSameLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        is_output_layer=True,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_successive_layers():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (10,)
    out_shape = (10, 8)
    
    X = np.random.randn(batch_size, in_shape[0], in_features)
    
    # Test normal case
    torch_model = TorchSuccessiveLayers(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    jax_model = JaxSuccessiveLayers(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)
    
    # Test last_dim_is_output case
    torch_model = TorchSuccessiveLayers(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        last_dim_is_output=True,
    )
    jax_model = JaxSuccessiveLayers(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        last_dim_is_output=True,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_non_neighbor_internal_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (10,)
    out_shape = (8, 10)
    
    X = np.random.randn(batch_size, in_shape[0], in_features)
    
    # Test normal case
    torch_model = TorchNonNeighborInternalLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    jax_model = JaxNonNeighborInternalLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)
    
    # Test first_dim_is_output case
    torch_model = TorchNonNeighborInternalLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        first_dim_is_output=True,
    )
    jax_model = JaxNonNeighborInternalLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        first_dim_is_output=True,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)
    
    # Test last_dim_is_input case
    torch_model = TorchNonNeighborInternalLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        last_dim_is_input=True,
    )
    jax_model = JaxNonNeighborInternalLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        last_dim_is_input=True,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_bias_to_weight_block():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    weight_shapes = ((8, 10), (10, 12), (12, 14))
    bias_shapes = ((10,), (12,), (14,))
    
    X = [np.random.randn(batch_size, shape[0], in_features) for shape in bias_shapes]
    
    # Test diagonal case
    torch_model = TorchBiasToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        diagonal=True,
    )
    jax_model = JaxBiasToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        diagonal=True,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, tuple(jnp.array(x) for x in X))
    
    # Forward pass
    torch_out = torch_model(tuple(torch.tensor(x) for x in X))
    torch_out = tuple(x.detach().numpy() for x in torch_out)
    jax_out = jax_model.apply(jax_params, tuple(jnp.array(x) for x in X))
    
    # Compare outputs
    for t_out, j_out in zip(torch_out, jax_out):
        np.testing.assert_allclose(t_out, j_out, rtol=1e-5, atol=1e-5)
    
    # Test full case
    torch_model = TorchBiasToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        diagonal=False,
    )
    jax_model = JaxBiasToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        diagonal=False,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, tuple(jnp.array(x) for x in X))
    
    # Forward pass
    torch_out = torch_model(tuple(torch.tensor(x) for x in X))
    torch_out = tuple(x.detach().numpy() for x in torch_out)
    jax_out = jax_model.apply(jax_params, tuple(jnp.array(x) for x in X))
    
    # Compare outputs
    for t_out, j_out in zip(torch_out, jax_out):
        np.testing.assert_allclose(t_out, j_out, rtol=1e-5, atol=1e-5) 