# -*- coding: utf-8 -*-
"""NODEProofofConcept.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J9TwyqIxWilb5VZwaojAiB9aXRbIBFnQ
"""

# This code cell installs packages on Colab

import sys
#if "google.colab" in sys.modules:
#    !wget "https://raw.githubusercontent.com/ndcbe/CBE60499/main/notebooks/helper.py"
#    import helper
#    helper.install_idaes()
#    helper.install_ipopt()
#    !pyomo build-extensions

#!pip install git+https://github.com/cog-imperial/OMLT.git

from omlt import OmltBlock, OffsetScaling
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
from omlt.io import load_keras_sequential

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Sample Problem 1 (Ex 1 from Dynopt Guide)
#
# 	min X2(tf)
# 	s.t.	X1_dot = u			X1(0) = 1
# 		X2_dot = X1^2 + u^2		X2(0) = 0
# 		tf = 1

from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()

m.t = ContinuousSet(bounds=(0, 1))

m.x1 = Var(m.t, bounds=(0, 1))
m.x2 = Var(m.t, bounds=(0, 1))
m.u = Var(m.t, initialize=0)

m.x1dot = DerivativeVar(m.x1)
m.x2dot = DerivativeVar(m.x2)

m.obj = Objective(expr=m.x2[1])


def _x1dot(M, i):
    if i == 0:
        return Constraint.Skip
    return M.x1dot[i] == M.u[i]


m.x1dotcon = Constraint(m.t, rule=_x1dot)


def _x2dot(M, i):
    if i == 0:
        return Constraint.Skip
    return M.x2dot[i] == M.x1[i] ** 2 + M.u[i] ** 2


m.x2dotcon = Constraint(m.t, rule=_x2dot)


def _init(M):
    yield M.x1[0] == 1
    yield M.x2[0] == 0
    yield ConstraintList.End


m.init_conditions = ConstraintList(rule=_init)

discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=10, ncp=1, scheme='LAGRANGE-RADAU')
#discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)

solver = SolverFactory('ipopt')

results = solver.solve(m, tee=True)

x1 = []
x2 = []
u = []
t = []

print(sorted(m.t))

for i in sorted(m.t):
    t.append(i)
    x1.append(value(m.x1[i]))
    x2.append(value(m.x2[i]))
    u.append(value(m.u[i]))

import matplotlib.pyplot as plt

plt.plot(t, x1)
plt.plot(t, x2)
plt.plot(t, u)
plt.xlabel('time')
plt.legend(['$x_1$','$x_2$','u'])
plt.show()

print(u)

"""NODE"""

import jax.numpy as jnp

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

import numpy.random as npr
from jax import jit, grad

resnet_depth = 3
def resnet_squared_loss(params, inputs, targets):
  preds = resnet(params, inputs, resnet_depth)
  return jnp.mean(jnp.sum((preds - targets)**2, axis=1))

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

# A simple gradient-descent optimizer.
@jit
def resnet_update(params, inputs, targets):
  grads = grad(resnet_squared_loss)(params, inputs, targets)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

inputy= jnp.transpose(jnp.array([x1,x2,u]))
inputyt= jnp.transpose(jnp.array([x1,x2,u,t]))
outputy= jnp.transpose(jnp.array([x1,x2,u]))

inputy.shape

# Toy 1D dataset.
#inputs = jnp.reshape(jnp.linspace(-2.0, 2.0, 10), (10, 1))
#targets = inputs**3 + 0.1 * inputs

# Hyperparameters.
layer_sizes = [3, 25, 3]
param_scale = 1.0
step_size = 0.01
train_iters = 1000

# Initialize and train.
resnet_params = init_random_params(param_scale, layer_sizes)
for i in range(train_iters):
  resnet_params = resnet_update(resnet_params, inputy, outputy)

# Plot results.
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 4), dpi=150)
ax = fig.gca()
ax.scatter(jnp.array(t), outputy[:,0], lw=0.5, color='green')
#ax.scatter(jnp.array(t), outputy[:,1], lw=0.5, color='green')
ax.scatter(jnp.array(t), outputy[:,2], lw=0.5, color='green')
#fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))
ax.plot(jnp.array(t), resnet(resnet_params, inputy, resnet_depth), lw=0.5, color='blue')
ax.set_xlabel('input')
ax.set_ylabel('output')

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

odenet_layer_sizes = [4, 20, 3]

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
  odenet_params = odenet_update(odenet_params, inputy, outputy)

# Plot resulting model.
fig = plt.figure(figsize=(6, 4), dpi=150)
ax = fig.gca()
ax.scatter(jnp.array(t), outputy[:,2], lw=0.5, color='green')
#fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))
#ax.plot(fine_inputs, resnet(resnet_params, fine_inputs, resnet_depth), lw=0.5, color='blue')
ax.plot(jnp.array(t), batched_odenet(odenet_params, inputy)[:,2], lw=0.5, color='red')
ax.set_xlabel('input')
ax.set_ylabel('output')
plt.legend(('True Data', 'ODE Net predictions'))

(odenet_params[0][0])

import tensorflow.keras as k

model=k.Sequential()
model.add(k.layers.Dense(20,activation='tanh', input_shape=(4,)))
model.add(k.layers.Dense(3))

model.set_weights([odenet_params[0][0],odenet_params[0][1],odenet_params[1][0],odenet_params[1][1]])

from omlt import OmltBlock, OffsetScaling
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
from omlt.io import load_keras_sequential

OM=ConcreteModel()
OM.nn=OmltBlock()

OM.x1=Var()
OM.x2=Var()
OM.u=Var()
OM.t=Var()
OM.dx1=Var()
OM.dx2=Var()
OM.du=Var()

net=load_keras_sequential(model)
formulation = FullSpaceNNFormulation(net)
OM.nn.build_formulation(formulation)

@OM.Constraint()
def connect_input(mdl):
    return mdl.x1 == mdl.nn.inputs[0]
@OM.Constraint()
def connect_output(mdl):
    return mdl.dx1 == mdl.nn.outputs[0]
@OM.Constraint()
def connect_input2(mdl):
    return mdl.x2 == mdl.nn.inputs[1]
@OM.Constraint()
def connect_output2(mdl):
    return mdl.dx2 == mdl.nn.outputs[1]
@OM.Constraint()
def connect_input3(mdl):
    return mdl.u == mdl.nn.inputs[2]
@OM.Constraint()
def connect_output3(mdl):
    return mdl.du == mdl.nn.outputs[2]
@OM.Constraint()
def connect_input4(mdl):
    return mdl.t == mdl.nn.inputs[3]

OM.obj = Objective(expr=(OM.x1 - 0.5)**2)
status = SolverFactory('ipopt').solve(OM, tee=False)
print(value(OM.x1))
print(value(OM.u))

"""Pyomo Example Optimal Control (2)"""

m=ConcreteModel()
m.tf=Param(initialize=1)
m.t =ContinuousSet(bounds=(0,m.tf))

m.u=Var(m.t,initialize=0)
m.x1=Var(m.t)
m.x2=Var(m.t)
m.x3=Var(m.t)

m.dx1dt = DerivativeVar(m.x1,wrt=m.t)
m.dx2dt = DerivativeVar(m.x2,wrt=m.t)
m.dx3dt = DerivativeVar(m.x3,wrt=m.t)

m.obj = Objective(expr=m.x3[m.tf])

def _x1dot(m,t):
  return m.dx1dt[t] == m.x2[t]
m.x1dot=Constraint(m.t, rule=_x1dot)
def _x2dot(m,t):
  return m.dx2dt[t] == -m.x2[t] +m.u[t]
m.x2dot=Constraint(m.t, rule=_x2dot)
def _x3dot(m,t):
  return m.dx3dt[t] == m.x1[t]**2 +m.x2[t]**2+0.005*m.u[t]**2
m.x3dot=Constraint(m.t, rule=_x3dot)

def _con(m,t):
  return m.x2[t]-8*(t-0.5)**2+0.5<=0
m.con = Constraint(m.t,rule=_con)

def _init(m):
  yield m.x1[0]==0
  yield m.x2[0]==-1
  yield m.x3[0]==0
m.init_conditions=ConstraintList(rule=_init)

discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=50, ncp=1, scheme='LAGRANGE-RADAU')
#discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)

solver = SolverFactory('ipopt')

results = solver.solve(m, tee=True)

x1 = []
x2 = []
x3 = []
u = []
t = []

print(sorted(m.t))

for i in sorted(m.t):
    t.append(i)
    x1.append(value(m.x1[i]))
    x2.append(value(m.x2[i]))
    x3.append(value(m.x3[i]))
    u.append(value(m.u[i]))

import matplotlib.pyplot as plt

plt.plot(t, x1)
plt.plot(t, x2)
plt.plot(t, x3)
plt.legend(['$x_1$','$x_2$','$x_3$'])
plt.show()

plt.plot(t, u,'r--')
plt.legend(['u'])
plt.show()

inputy= jnp.transpose(jnp.array([x1,x2,x3,u]))
inputyt= jnp.transpose(jnp.array([x1,x2,x3,u,t]))
outputy= jnp.transpose(jnp.array([x1,x2,x3]))

odenet_params[1][1].shape

# We need to change the input dimension to 2, to allow time-dependent dynamics.
odenet_layer_sizes = [5, 40, 3]

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
train_iters=3000
for i in range(train_iters):
  odenet_params = odenet_update(odenet_params, inputy, outputy)

# Plot resulting model.
fig = plt.figure(figsize=(6, 4), dpi=150)
ax = fig.gca()
ax.scatter(jnp.array(t), outputy[:,0], lw=0.5, color='green')
#fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))
#ax.plot(fine_inputs, resnet(resnet_params, fine_inputs, resnet_depth), lw=0.5, color='blue')
ax.plot(jnp.array(t), batched_odenet(odenet_params, inputy)[:,0], lw=0.5, color='red')
ax.set_xlabel('input')
ax.set_ylabel('output')
plt.legend(('True Data', 'ODE Net predictions'))

