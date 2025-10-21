.. _api_ref:

.. currentmodule:: nsbtools

API Reference
=============

.. contents:: **List of modules**
   :local:

.. _ref_eigen:

:mod:`nsbtools.eigen` - Eigenmode analyses on cortical surfaces
---------------------------------------------------------------

.. automodule:: nsbtools.eigen
   :no-members:
   :no-inherited-members:

.. currentmodule:: nsbtools.eigen

.. autosummary::
   :template: class.rst
   :toctree: generated/

   EigenSolver

.. autoclass:: EigenSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nsbtools.eigen.EigenSolver.solve
   nsbtools.eigen.EigenSolver.decompose
   nsbtools.eigen.EigenSolver.reconstruct
   nsbtools.eigen.EigenSolver.simulate_waves

.. autosummary::
   :template: function.rst
   :toctree: generated/

   check_surf
   mask_surf
   check_hetero
   scale_hetero
   check_orthonorm_modes
   standardize_modes
   gen_random_input
   model_wave_fourier
   solve_wave_ode
   model_balloon_fourier
   model_balloon_ode