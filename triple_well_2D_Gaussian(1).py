#!/usr/bin/env python
# coding: utf-8

# To demonstrate EDMD and deeptime's API, we consider a two-dimensional triple-well potential. it is defined on $[-2, 2]\times [-1, 2]$ and looks like the following:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
from deeptime.basis import Gaussian, Monomials, Rinn, Genbasis
np_config.enable_numpy_behavior()
from deeptime.data import triple_well_2d
system = triple_well_2d()

x = np.linspace(-2, 2, num=100)
y = np.linspace(-1, 2, num=100)
XX, YY = np.meshgrid(x, y)
coords = np.dstack((XX, YY)).reshape(-1, 2)
V = system.potential(coords).reshape(XX.shape)

plt.contourf(x, y, V, levels=np.linspace(-4.5, 4.5, 20), cmap='coolwarm');


# Inside this potential landscape we integrate particles subject to the stochastic differential equation
# 
# $$dX_t = \nabla V(X_t)dt + \sigma(t, X_t)dW_t$$
# with $\sigma =1.09$, using an Euler-Maruyama integrator.

# In[2]:


traj = system.trajectory(x0=[[-1, 0]], length=200, seed=66)

plt.contourf(x, y, V, levels=np.linspace(-4.5, 4.5, 20), cmap='coolwarm')
plt.plot(*traj.T, c='black', lw=.5);


# To approximate the Koopman operator, we first select 15000 random points inside the domain and integrate them for 10000 steps under an integration step of $h=10^{−5}$:

# In[3]:


N = 15000
state = np.random.RandomState(seed=42)
X = np.stack([state.uniform(-2, 2, size=(N,)), state.uniform(-1, 2, size=(N,))]).T
Y = system(X, n_jobs=8)


# Now we pick a basis to be all monomials up to and including degree deg=10.

# In[5]:


from deeptime.decomposition import EDMD
from scipy import linalg
from numpy.linalg import norm 


# First by using Gaussians as a basis functions
M = 50
rho = 1.05
centers = np.stack([np.random.uniform(-2, 2, size=(M,)), np.random.uniform(-1, 2, size=(M,))]).T
gbasis = Genbasis("Gaussianbasis", M, centers, rho)
psis = gbasis._evaluate(X)

# Second by using neural networks as a basis functions

#M = 10 # number of basis functions
#units = [4,1]
#activations = ["relu","sigmoid"]
#nbasis = Genbasis("Rinnbasis", M, units, activations)
#psis = nbasis._evaluate(X)


# Third by using Monomials basis functions

#mbasis = Monomials(p=10, d=2)
#mbasis._evaluate(X)


# Visualize the first four basis functions

# In[6]:



fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(ncols=2, nrows=2)

ax = fig.add_subplot(gs[0, 0],projection='3d')
s = ax.scatter(*X.T, psis[:, 0])
#ax.set_ylim(-2,0)
ax.set_title('1st basis function')
#fig.colorbar(s)

ax = fig.add_subplot(gs[0, 1],projection='3d')
s = ax.scatter(*X.T, psis[:, 1])
ax.set_title('2nd basis function')
#fig.colorbar(s)
ax = fig.add_subplot(gs[1, 0],projection='3d')
s = ax.scatter(*X.T, psis[:, 2])
ax.set_title('3rd basis function')
#fig.colorbar(s)
ax = fig.add_subplot(gs[1, 1],projection='3d')
s = ax.scatter(*X.T, psis[:, 3])
ax.set_title('4th basis function')
#fig.colorbar(s)


# Using this basis, we can fit an EDMD estimator and obtain the corresponding model.

# In[7]:


from deeptime.decomposition import EDMD
# Combining EDMD with the required basis used gbasis or nbasis or mbasis
edmd_estimator = EDMD(gbasis, n_eigs=8)
edmd_model = edmd_estimator.fit((X, Y)).fetch_model()


# We can obtain the dominant eigenvalues associated to processes in the system.

# In[8]:


plt.plot(np.real(edmd_model.eigenvalues), 'x')
plt.title('Dominant eigenvalues');


# In[9]:


edmd_model.eigenvalues


# We project our input data X using the eigenfunctions:

# In[10]:


phi = np.real(edmd_model.transform(X, propagate=False))

# normalize
for i in range(len(edmd_model.eigenvalues)):
    phi[:, i] = phi[:, i] / np.max(np.abs(phi[:, i]))


# And visualize the first four eigenfunctions. They contain information about metastable regions and slow transitions.

# In[11]:


fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(ncols=2, nrows=2)

ax = fig.add_subplot(gs[0, 0])
s = ax.scatter(*X.T, c= phi[:, 0], cmap='coolwarm')
#ax.set_ylim(0,2)
ax.set_title('1st eigenfunction')
fig.colorbar(s)


ax = fig.add_subplot(gs[0, 1])
s = ax.scatter(*X.T, c= phi[:, 1], cmap='coolwarm')
ax.set_title('2nd eigenfunction')
fig.colorbar(s)

ax = fig.add_subplot(gs[1, 0])
s = ax.scatter(*X.T, c= phi[:, 2], cmap='coolwarm')
ax.set_title('3rd eigenfunction')
fig.colorbar(s)

ax = fig.add_subplot(gs[1, 1])
s = ax.scatter(*X.T, c= phi[:, 3], cmap='coolwarm')
#ax.set_ylim(-2,0)
ax.set_title('4th eigenfunction')
fig.colorbar(s)


# Using a clustering can reveal temporally coherent structures.

# In[12]:


from deeptime.clustering import KMeans

kmeans = KMeans(n_clusters=6, n_jobs=4).fit(np.real(phi)).fetch_model()
c = kmeans.transform(np.real(phi))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.scatter(*X.T, c=c)
ax1.set_title(r"$t = 0$")
ax1.set_aspect('equal')
ax1.set_xlim([-2.5, 2.5])
ax1.set_ylim([-1.5, 2.5])
ax1.contour(x, y, V, levels=np.linspace(-4.5, 4.5, 20), colors='black')

ax2.scatter(*Y.T, c=c)
ax2.set_title(r"$t = 0.1$")
ax2.set_aspect('equal')
ax2.set_xlim([-2.5, 2.5])
ax2.set_ylim([-1.5, 2.5])
ax2.contour(x, y, V, levels=np.linspace(-4.5, 4.5, 20), colors='black');


# In[ ]:





# In[ ]:





# In[ ]:




