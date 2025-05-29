import jax
import jax.numpy as jnp
import numpy as np
import torch
import random
from nn.models import (
    MLPModel as TorchMLPModel,
    MLPModelForClassification as TorchMLPModelForClassification,
    DWSModel as TorchDWSModel,
    DWSModelForClassification as TorchDWSModelForClassification,
)
from nn.models_jax import (
    MLPModel as JaxMLPModel,
    MLPModelForClassification as JaxMLPModelForClassification,
    DWSModel as JaxDWSModel,
    DWSModelForClassification as JaxDWSModelForClassification,
)

torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def test_mlp_model():
    # Test data
    batch_size = 4
    in_dim = 2090
    hidden_dim = 256
    n_hidden = 2
    
    # Create random input data
    weight_shapes = [(10, 20), (20, 30), (30, 40)]
    bias_shapes = [(20,), (30,), (40,)]
    
    weights = [np.random.randn(batch_size, *shape, 1) for shape in weight_shapes]
    biases = [np.random.randn(batch_size, *shape, 1) for shape in bias_shapes]
    X = (weights, biases)
    
    # Initialize models
    torch_model = TorchMLPModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    )
    jax_model = JaxMLPModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, X)
    
    # Forward pass
    X_torch = (
        tuple(torch.tensor(x) for x in X[0]),
        tuple(torch.tensor(x) for x in X[1]),
    )
    torch_out = torch_model(X_torch)
    jax_out = jax_model.apply(jax_params, X)
    
    # Compare outputs
    for t_out, j_out in zip(torch_out[0], jax_out[0]):
        for t, j in zip(t_out, j_out):
            np.testing.assert_allclose(t.detach().numpy(), j, rtol=1e-5, atol=1e-5)
    for t_out, j_out in zip(torch_out[1], jax_out[1]):
        for t, j in zip(t_out, j_out):
            np.testing.assert_allclose(t.detach().numpy(), j, rtol=1e-5, atol=1e-5)


def test_mlp_model_for_classification():
    # Test data
    batch_size = 4
    in_dim = 2090
    hidden_dim = 256
    n_hidden = 2
    n_classes = 10
    
    # Create random input data
    weight_shapes = [(10, 20), (20, 30), (30, 40)]
    bias_shapes = [(20,), (30,), (40,)]
    
    X = (
        tuple(np.random.randn(batch_size, *shape, 1) for shape in weight_shapes),
        tuple(np.random.randn(batch_size, *shape, 1) for shape in bias_shapes),
    )
    
    # Initialize models
    torch_model = TorchMLPModelForClassification(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        n_classes=n_classes,
    )
    jax_model = JaxMLPModelForClassification(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        n_classes=n_classes,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, X)

    X_torch = (
        tuple(torch.tensor(x) for x in X[0]),
        tuple(torch.tensor(x) for x in X[1]),
    )
    
    # Forward pass
    torch_out = torch_model(X_torch)
    jax_out = jax_model.apply(jax_params, X)
    
    # Compare outputs
    np.testing.assert_allclose(torch_out.detach().numpy(), jax_out, rtol=1e-5, atol=1e-5)


def test_dws_model():
    # Test data
    batch_size = 4
    input_features = 2
    hidden_dim = 16
    n_hidden = 2
    
    # Create random input data
    weight_shapes = [(10, 20), (20, 30), (30, 40)]
    bias_shapes = [(20,), (30,), (40,)]
    
    X = (
        tuple(np.random.randn(batch_size, *shape, input_features) for shape in weight_shapes),
        tuple(np.random.randn(batch_size, *shape, input_features) for shape in bias_shapes),
    )

    X_torch = (
        tuple(torch.tensor(x) for x in X[0]),
        tuple(torch.tensor(x) for x in X[1]),
    )
    
    # Initialize models
    torch_model = TorchDWSModel(
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=input_features,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    )
    jax_model = JaxDWSModel(
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=input_features,
        output_features=hidden_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, X)
    
    # Forward pass
    torch_out = torch_model(X_torch)
    jax_out = jax_model.apply(jax_params, X)
    
    # Compare outputs
    for t_out, j_out in zip(torch_out[0], jax_out[0]):
        for t, j in zip(t_out, j_out):
            np.testing.assert_allclose(t.detach().numpy(), j, rtol=1e-5, atol=1e-5)
    for t_out, j_out in zip(torch_out[1], jax_out[1]):
        for t, j in zip(t_out, j_out):
            np.testing.assert_allclose(t.detach().numpy(), j, rtol=1e-5, atol=1e-5)


def test_dws_model_for_classification():
    # Test data
    batch_size = 4
    input_features = 2
    hidden_dim = 16
    n_hidden = 2
    n_classes = 10
    
    # Create random input data
    weight_shapes = [(10, 20), (20, 30), (30, 40)]
    bias_shapes = [(20,), (30,), (40,)]
    
    X = (
        tuple(np.random.randn(batch_size, *shape, input_features) for shape in weight_shapes),
        tuple(np.random.randn(batch_size, *shape, input_features) for shape in bias_shapes),
    )

    X_torch = (
        tuple(torch.tensor(x) for x in X[0]),
        tuple(torch.tensor(x) for x in X[1]),
    )
    
    # Initialize models
    torch_model = TorchDWSModelForClassification(
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=input_features,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        n_classes=n_classes,
    )
    jax_model = JaxDWSModelForClassification(
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=input_features,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        n_classes=n_classes,
    )
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, X)
    
    # Forward pass
    torch_out = torch_model(X_torch)
    jax_out = jax_model.apply(jax_params, X)
    
    # Compare outputs
    np.testing.assert_allclose(torch_out.detach().numpy(), jax_out, rtol=1e-5, atol=1e-5)