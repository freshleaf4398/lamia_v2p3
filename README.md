# BEM with surface integral equation (SIE) method (Lamia v2.3) 

LAMIA = Light Absorption and Multinanoparticle Interaction Algorithm 

Clean Python version from https://github.com/freshleaf4398/Surface-Integral-Equation/tree/main. SALOME med mesh files operations are added, instead of COMSOL mphtxt/mat mesh files. Finding the farfield distribution, T-matrix, multipole decomposition, and cross sections from a homogeneous nanoparticle systems. 

The radius of the farfield should not be very high, the code can mess the ext and abs cross section.

Latest features: 
* Nearfield distributions with masks 
* One cell runs
* Add global T-matrix algorithm
* Helicity basis
