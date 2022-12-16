# What is RDMD?
RDMD is the abbreviation of using EDMD with random basis functions, e.g., EDMD combined with random gaussians (by choosing random centers) or randomly initialised neural network as a basis functions to approximate the Koopman operator.

# What is Deeptime

Deeptime is a general purpose Python library offering various tools to estimate dynamical models based on time-series data including conventional linear learning methods, such as Markov State Models (MSMs), Hidden Markov Models (HMMs) and Koopman models, as well as kernel and deep learning approaches such as VAMPnets and deep MSMs. The library is largely compatible with scikit-learn, having a range of Estimator classes for these different models, but in contrast to scikit-learn also provides Model classes, e.g., in the case of an MSM, which provide a multitude of analysis methods to compute interesting thermodynamic, kinetic and dynamical quantities, such as free energies, relaxation times and transition paths.

Deeptime can be installed via conda (conda install -c conda-forge deeptime) delivering pre-compiled binaries and is also available via pip (pip install deeptime), causing the binaries to be compiled locally. 
