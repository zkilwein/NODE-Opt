import pandas as pd
import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad

df_t=pd.read_csv('t.csv')
df_u=pd.read_csv('u.csv')
df_x1=pd.read_csv('X1.csv')
df_x2=pd.read_csv('X2.csv')
u_full=df_u.to_numpy()[:,1:].reshape(-1)[0:100]
t_full=df_t.to_numpy()[:,1:].reshape(-1)[0:100]
x1_full=df_x1.to_numpy()[:,1:].reshape(-1)[0:100]
x2_full=df_x2.to_numpy()[:,1:].reshape(-1)[0:100]

input_full= jnp.transpose(jnp.array([x1_full,x2_full,u_full]))
input_full_t= jnp.transpose(jnp.array([x1_full,x2_full,u_full,t_full]))
output_full= jnp.transpose(jnp.array([x1_full,x2_full,u_full]))

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]
def relu(x):
  return jnp.maximum(0.0, x)

def mlp(params, inputs):
  # A multi-layer perceptron, i.e. a fully-connected neural network.
  for w, b in params:
    outputs = jnp.dot(inputs, w) + b  # Linear transform
    inputs = jnp.tanh(outputs)            # Nonlinearity
  return outputs
def resnet(params, inputs, depth):
  for i in range(depth):
    outputs = mlp(params, inputs) + inputs
  return outputs

def nn_dynamics(state, time, params):
  state_and_time = jnp.hstack([state, jnp.array(time)])
  return mlp(params, state_and_time)

from jax.experimental.ode import odeint

def odenet(params, input):
  start_and_end_times = jnp.array([0.0, 1.0])
  init_state, final_state = odeint(nn_dynamics, input, start_and_end_times, params, atol=0.001, rtol=0.001)
  return final_state
from jax import vmap
batched_odenet = vmap(odenet, in_axes=(None, 0))

odenet_layer_sizes = [4, 50, 3]
# Hyperparameters.
layer_sizes = [3, 25, 3]
param_scale = 1.0
step_size = 0.01
train_iters = 200
def odenet_loss(params, inputs, targets):
  preds = batched_odenet(params, inputs)
  return jnp.mean(jnp.sum((preds - targets)**2, axis=1))

@jit
def odenet_update(params, inputs, targets):
  grads = grad(odenet_loss)(params, inputs, targets)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

# Initialize and train ODE-Net.
odenet_params = init_random_params(param_scale, odenet_layer_sizes)

for i in range(train_iters):
  odenet_params = odenet_update(odenet_params, input_full, output_full)


import matplotlib.pyplot as plt

# Plot resulting model.
fig = plt.figure(figsize=(6, 4), dpi=150)
ax = fig.gca()
ax.scatter(t_full[0:100], x1_full[0:100], lw=0.5, color='green')
#fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))
#ax.plot(fine_inputs, resnet(resnet_params, fine_inputs, resnet_depth), lw=0.5, color='blue')
ax.plot(t_full[0:100], batched_odenet(odenet_params, input_full)[0:100,0], lw=0.5, color='red')
ax.set_xlabel('input')
ax.set_ylabel('output')
plt.legend(('True Data', 'ODE Net predictions'))
fig.savefig('Test2.png')
