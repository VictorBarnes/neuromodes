.. image:: https://github.com/izachp/nsbtools/raw/main/docs/_static/logo.png
   :alt: Design by Gabriella Chan

Eigenmode-based neuroimaging tools developed by the `Neural Systems and Behaviour Lab <https://www.monash.edu/medicine/psych/alex-fornito-lab>`_.

Features
--------
- Compute geometric eigenmodes from cortical surface meshes
- Decompose and reconstruct cortical maps using an orthogonal basis set
- Simulate neural and BOLD activity using neural field theory
- Generate structural connectomes using the method from `Normand et al. (2025) <https://doi.org/10.1101/2025.09.17.676944>`_
- Incorporate spatial heterogeneity into all of the above via cortical maps
- Visualise matrices and cortical maps
- Access triangular surface meshes for human, macaque, and marmoset cortices

All human cortical surface meshes provided were obtained from the `neuromaps <https://neuromaps-main.readthedocs.io/en/stable/index.html>`_ package, which also provides a diverse set of cortical maps that can be used for our `EigenSolver`'s `hetero` parameter.

To compare cortical maps while accounting for spatial autocorrelation, we recommend using the `eigenstrapping <https://eigenstrapping.readthedocs.io/en/stable/index.html>`_ package, which implements a rigorous null model based on geometric eigenmodes.

Installation
------------
``nsbtools`` works with Python 3.9 to 3.12. It can be installed by cloning the repository and installing from the local directory:

::

  git clone https://github.com/NSBLab/nsbtools
  cd nsbtools
  pip install .

Tests can be run with ``pytest``:

::

  cd nsbtools
  pip install pytest
  pytest tests

Alternatively, ``nsbtools`` can be installed via `UV <https://docs.astral.sh/uv/>`_ by running ``uv sync`` from the repository's root or ``uv add <path/to/nsbtools>`` from another project's directory.

Citing
------
If you use ``nsbtools`` in your work, please cite the following three papers:

J.C. Pang, K.M. Aquino, M. Oldehinkel, P.A. Robinson, B.D. Fulcher, M. Breakspear, A. Fornito, Geometric constraints on human brain function, Nature, 618, 566–574 (2023) (DOI: 10.1038/s41586-023-06098-1)

M\. Reuter, F-E. Wolter, N. Peinecke, Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and solids, Computer-Aided Design, 38(4), 342-366 (2006). (DOI: 10.1016/j.cad.2005.10.011)

C\. Wachinger, P. Golland, W. Kremen, B. Fischl, M. Reuter, BrainPrint: a discriminative characterization of brain morphology, Neuroimage, 109, 232-248 (2015). (DOI: 10.1016/j.neuroimage.2015.01.032)

Additionally, if you use the ``generate_connectome`` function, please also cite:

F\. Normand, M. Gajwani, T. Cao, J. Cruddas, A. Sangchooli, S. Oldham, A. Holmes, P.A. Robinson, J.C. Pang, A. Fornito, Geometric constraints on the architecture of mammalian cortical connectomes, BioRxiv (2025) (DOI: 10.1101/2025.09.17.676944)

License information
-------------------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (``cc-by-nc-sa``). See the `LICENSE <LICENCE-CC-BY-NC-SA-4.0.md>`_ file for details.
