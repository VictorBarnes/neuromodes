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

   nsbtools.eigen.EigenSolver.compute_lbo
   nsbtools.eigen.EigenSolver.solve
   nsbtools.eigen.EigenSolver.decompose
   nsbtools.eigen.EigenSolver.reconstruct
   nsbtools.eigen.EigenSolver.generate_connectome
   nsbtools.eigen.EigenSolver.simulate_waves

.. autosummary::
   :template: function.rst
   :toctree: generated/

   is_valid_hetero
   scale_hetero
   is_mass_orthonormal_modes
   standardize_modes
   calc_norm_power
   decompose
   reconstruct

.. _ref_waves:

:mod:`nsbtools.waves` - Run the simple wave model
---------------------------------------------------------------

.. automodule:: nsbtools.waves
   :no-members:
   :no-inherited-members:

.. currentmodule:: nsbtools.waves

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nsbtools.waves.simulate_waves

.. _ref_connectome:

:mod:`nsbtools.connectome` - Generative network modelling
---------------------------------------------------------------

.. automodule:: nsbtools.connectome
   :no-members:
   :no-inherited-members:

.. currentmodule:: nsbtools.connectome

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nsbtools.connectome.generate_connectome

.. _ref_io:

:mod:`nsbtools.io` - IO functions for cortical surface meshes and maps
---------------------------------------------------------------

.. automodule:: nsbtools.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: nsbtools.io

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nsbtools.io.read_surf
   nsbtools.io.mask_surf
   nsbtools.io.check_surf
   nsbtools.io.fetch_surf
   nsbtools.io.fetch_map