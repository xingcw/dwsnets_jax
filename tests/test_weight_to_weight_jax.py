import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from nn.layers.weight_to_weight import (
    GeneralMatrixSetLayer as TorchGeneralMatrixSetLayer,
    SetKroneckerSetLayer as TorchSetKroneckerSetLayer,
    FromFirstLayer as TorchFromFirstLayer,
    ToFirstLayer as TorchToFirstLayer,
    FromLastLayer as TorchFromLastLayer,
    ToLastLayer as TorchToLastLayer,
    NonNeighborInternalLayer as TorchNonNeighborInternalLayer,
    WeightToWeightBlock as TorchWeightToWeightBlock,
)
from nn.layers_jax.weight_to_weight_jax import (
    GeneralMatrixSetLayer as JaxGeneralMatrixSetLayer,
    SetKroneckerSetLayer as JaxSetKroneckerSetLayer,
    FromFirstLayer as JaxFromFirstLayer,
    ToFirstLayer as JaxToFirstLayer,
    FromLastLayer as JaxFromLastLayer,
    ToLastLayer as JaxToLastLayer,
    NonNeighborInternalLayer as JaxNonNeighborInternalLayer,
    WeightToWeightBlock as JaxWeightToWeightBlock,
)

def test_general_matrix_set_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (8, 10)
    out_shape = (10, 12)
    
    X = np.random.randn(batch_size, in_shape[0], in_shape[1], in_features)
    
    # Test same index case
    torch_model = TorchGeneralMatrixSetLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        in_index=0,
        out_index=0,
    )
    jax_model = JaxGeneralMatrixSetLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        in_index=0,
        out_index=0,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)
    
    # Test consecutive indices case
    torch_model = TorchGeneralMatrixSetLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        in_index=0,
        out_index=1,
    )
    jax_model = JaxGeneralMatrixSetLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        in_index=0,
        out_index=1,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_set_kronecker_set_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (8, 10)
    
    X = np.random.randn(batch_size, in_shape[0], in_shape[1], in_features)
    
    torch_model = TorchSetKroneckerSetLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
    )
    jax_model = JaxSetKroneckerSetLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_from_first_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (8, 10)
    out_shape = (10, 12)
    
    X = np.random.randn(batch_size, in_shape[0], in_shape[1], in_features)
    
    # Test normal case
    torch_model = TorchFromFirstLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    jax_model = JaxFromFirstLayer(
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
    torch_model = TorchFromFirstLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        last_dim_is_output=True,
    )
    jax_model = JaxFromFirstLayer(
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

def test_to_first_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (8, 10)
    out_shape = (10, 12)
    
    X = np.random.randn(batch_size, in_shape[0], in_shape[1], in_features)
    
    # Test normal case
    torch_model = TorchToFirstLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    jax_model = JaxToFirstLayer(
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
    torch_model = TorchToFirstLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
        first_dim_is_output=True,
    )
    jax_model = JaxToFirstLayer(
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

def test_from_last_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (8, 10)
    out_shape = (10, 12)
    
    X = np.random.randn(batch_size, in_shape[0], in_shape[1], in_features)
    
    torch_model = TorchFromLastLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    jax_model = JaxFromLastLayer(
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

def test_to_last_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (8, 10)
    out_shape = (10, 12)
    
    X = np.random.randn(batch_size, in_shape[0], in_shape[1], in_features)
    
    torch_model = TorchToLastLayer(
        in_features=in_features,
        out_features=out_features,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    jax_model = JaxToLastLayer(
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

def test_non_neighbor_internal_layer():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    in_shape = (8, 10)
    out_shape = (10, 12)
    
    X = np.random.randn(batch_size, in_shape[0], in_shape[1], in_features)
    
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

def test_weight_to_weight_block():
    # Test data
    batch_size = 4
    in_features = 32
    out_features = 64
    shapes = ((8, 10), (10, 12), (12, 14))
    
    X = [np.random.randn(batch_size, shape[0], shape[1], in_features) for shape in shapes]
    
    # Test diagonal case
    torch_model = TorchWeightToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        shapes=shapes,
        diagonal=True,
    )
    jax_model = JaxWeightToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        shapes=shapes,
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
    torch_model = TorchWeightToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        shapes=shapes,
        diagonal=False,
    )
    jax_model = JaxWeightToWeightBlock(
        in_features=in_features,
        out_features=out_features,
        shapes=shapes,
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