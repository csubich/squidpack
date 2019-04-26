# squidpack
Spherical Quadrature on Icosahedral Domains

This repository contains the code for SQUIDpack, first presented at the 2019 PDEs on the Sphere conference [held](http://collaboration.cmc.ec.gc.ca/science/pdes-2019/index-en.html) in Montréal, Québec.

The included python modules are:

* `quadfuncs.py` -- computes a normalized linear combination of (`x`, `y`, and `z`) monomials, such that the resulting functions are approximately orthogonal over a small region near the north pole of a sphere.

* `calcquad.py` -- Computes a quadrature rule for a grid of spherical hexagons/pentagons

* `prec_quadrature.ipynb` -- A Jupyter notebook which uses the above modules to compute quadrature rules for a selection of progressively refined icosahedral grids; it generates the figures used in the aformentioned presentation.

The core Python modules require:

* [Numpy](https://www.numpy.org), for high-performance Python numerical arrays
* [Sympy](https://www.sympy.org), for symbolic/algebraic routines used to generate the orthogonal moment functions
* [Mpmath](http://mpmath.org), for high-precision floating point objects used in some intermediate calculations

And the Jupyter notebook additionally requires:

* [Scipy](https://www.scipy.org), for its `spatial` submodule used to generate and manipulate the convex hull given scattered gridpoints
* [iModel](https://github.com/pedrospeixoto/iModel) by Pedro Peixoto, for the repository of pre-generated grids used (see its `getgridsfromserver.sh`)
* [ipyparallel](https://github.com/ipython/ipyparallel), for parallel processing of elements (it is optionally used by `calcquad`).  This dependency could be made optional with minor changes to the notebook.
