import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from nn.inr import INR as TorchINR
from nn.inr_jax import INR as JaxINR

jax.config.update("jax_enable_x64", True)


def test_sine_activation():
    # Test data
    x = np.random.randn(10, 2)
    x_torch = torch.tensor(x, dtype=torch.float32)
    x_jax = jnp.array(x, dtype=jnp.float32)

    # Initialize models
    torch_model = TorchINR(in_dim=2, n_layers=3, up_scale=4)
    jax_model = JaxINR(in_dim=2, n_layers=3, up_scale=4)

    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, x_jax)

    # Forward pass
    torch_out = torch_model(x_torch).detach().numpy()
    jax_out = jax_model.apply(jax_params, x_jax)

    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_positional_encoding():
    # Test data
    x = np.random.randn(10, 2)
    x_torch = torch.tensor(x, dtype=torch.float32)
    x_jax = jnp.array(x)

    # Initialize models with positional encoding
    torch_model = TorchINR(in_dim=2, n_layers=3, up_scale=4, pe_features=2)
    jax_model = JaxINR(in_dim=2, n_layers=3, up_scale=4, pe_features=2)

    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, x_jax)

    # Forward pass
    torch_out = torch_model(x_torch).detach().numpy()
    jax_out = jax_model.apply(jax_params, x_jax)

    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-4, atol=1e-4)

def test_gaussian_encoding():
    # Test data
    x = np.random.randn(10, 2)
    x_torch = torch.tensor(x, dtype=torch.float32)
    x_jax = jnp.array(x, dtype=jnp.float32)

    # Initialize models with Gaussian encoding
    torch_model = TorchINR(in_dim=2, n_layers=3, up_scale=4, fix_pe=False)
    jax_model = JaxINR(in_dim=2, n_layers=3, up_scale=4, fix_pe=False)

    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, x_jax)

    # Forward pass
    torch_out = torch_model(x_torch).detach().numpy()
    jax_out = jax_model.apply(jax_params, x_jax, rngs={'params': key})

    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_model_shapes():
    # Test different input shapes
    batch_sizes = [1, 4, 8]
    input_dims = [2, 3, 4]
    
    for batch_size in batch_sizes:
        for input_dim in input_dims:
            # Test data
            x = np.random.randn(batch_size, input_dim)
            x_jax = jnp.array(x)
            
            # Initialize model
            model = JaxINR(in_dim=input_dim, n_layers=3, up_scale=4)
            
            # Initialize parameters
            key = jax.random.PRNGKey(0)
            params = model.init(key, x_jax)
            
            # Forward pass
            out = model.apply(params, x_jax)
            
            # Check output shape
            assert out.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {out.shape}"

def test_gradient_flow():
    # Test that gradients flow through the model
    x = jnp.array(np.random.randn(10, 2))
    
    # Initialize model
    model = JaxINR(in_dim=2, n_layers=3, up_scale=4)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = model.init({'params': key}, x)
    
    # Define loss function
    def loss_fn(params, x):
        out = model.apply(params, x)
        return jnp.mean(out ** 2)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, x)
    
    # Check that gradients are not None
    assert grads is not None, "Gradients should not be None"
    
    # Check that gradients are not all zero
    has_nonzero_grad = False
    for param_grad in jax.tree_util.tree_leaves(grads):
        if jnp.any(param_grad != 0):
            has_nonzero_grad = True
            break
    assert has_nonzero_grad, "Some gradients should be non-zero" 