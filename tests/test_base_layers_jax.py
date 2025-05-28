import jax
import jax.numpy as jnp
import numpy as np
import torch
import random
import pytest
from nn.layers.base import BaseLayer as TorchBaseLayer
from nn.layers.base import MAB as TorchMAB
from nn.layers.base import SAB as TorchSAB
from nn.layers.base import SetLayer as TorchSetLayer
from nn.layers.base import GeneralSetLayer as TorchGeneralSetLayer
from nn.layers.base import Attn as TorchAttn
from nn.layers_jax.base_jax import BaseLayer as JaxBaseLayer
from nn.layers_jax.base_jax import MAB as JaxMAB
from nn.layers_jax.base_jax import SAB as JaxSAB
from nn.layers_jax.base_jax import SetLayer as JaxSetLayer
from nn.layers_jax.base_jax import GeneralSetLayer as JaxGeneralSetLayer
from nn.layers_jax.base_jax import Attn as JaxAttn

torch.set_default_dtype(torch.float64)
jax.config.update("jax_enable_x64", True)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def test_mab():
    # Test data
    batch_size = 2
    set_size = 2
    dim_Q = 8
    dim_K = 8
    dim_V = 8
    num_heads = 2
    
    Q = np.random.randn(batch_size, set_size, dim_Q).astype(np.float64)
    K = np.random.randn(batch_size, set_size, dim_K).astype(np.float64)
    
    # Initialize models
    torch_model = TorchMAB(dim_Q, dim_K, dim_V, num_heads)
    jax_model = JaxMAB(dim_Q=dim_Q, dim_K=dim_K, dim_V=dim_V, num_heads=num_heads)
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(Q), jnp.array(K))
    
    # Forward pass
    torch_out = torch_model(torch.from_numpy(Q), torch.from_numpy(K)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(Q), jnp.array(K))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_sab():
    # Test data
    batch_size = 4
    set_size = 2
    in_features = 8
    out_features = 8
    num_heads = 2
    
    X = np.random.randn(batch_size, set_size, in_features)
    
    # Initialize models
    torch_model = TorchSAB(in_features, out_features, num_heads)
    jax_model = JaxSAB(in_features=in_features, out_features=out_features, num_heads=num_heads)
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_set_layer():
    # Test data
    batch_size = 4
    set_size = 2
    in_features = 8
    out_features = 8
    
    X = np.random.randn(batch_size, set_size, in_features)  # 10 is the set size
    
    # Initialize models
    torch_model = TorchSetLayer(in_features, out_features)
    jax_model = JaxSetLayer(in_features=in_features, out_features=out_features)
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.tensor(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_general_set_layer():
    # Test data
    batch_size = 4  
    set_size = 2
    in_features = 8
    out_features = 8
    
    X = np.random.randn(batch_size, set_size, in_features)  # 10 is the set size
    
    # Test both "ds" and "sab" set layers
    for set_layer in ["ds", "sab"]:
        # Initialize models
        torch_model = TorchGeneralSetLayer(in_features, out_features, set_layer=set_layer)
        jax_model = JaxGeneralSetLayer(in_features=in_features, out_features=out_features, set_layer=set_layer)
        
        # Initialize JAX model parameters
        key = jax.random.PRNGKey(0)
        jax_params = jax_model.init(key, jnp.array(X))
        
        # Forward pass
        torch_out = torch_model(torch.tensor(X)).detach().numpy()
        jax_out = jax_model.apply(jax_params, jnp.array(X))
        
        # Compare outputs
        np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5)

def test_attn():
    # Test data
    batch_size = 4
    set_size = 2
    dim = 8
    
    X = np.random.randn(batch_size, dim, set_size)
    
    # Initialize models
    torch_model = TorchAttn(dim)
    jax_model = JaxAttn(dim=dim)
    
    # Initialize JAX model parameters
    key = jax.random.PRNGKey(0)
    jax_params = jax_model.init(key, jnp.array(X))
    
    # Forward pass
    torch_out = torch_model(torch.from_numpy(X)).detach().numpy()
    jax_out = jax_model.apply(jax_params, jnp.array(X))
    
    # Compare outputs
    np.testing.assert_allclose(torch_out, jax_out, rtol=1e-5, atol=1e-5) 



if __name__ == "__main__":
    test_attn()