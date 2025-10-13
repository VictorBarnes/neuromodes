.. image:: https://github.com/izachp/nsbtools/raw/main/docs/_static/logo.png
   :alt: Design by Gabriella Chan

Eigenmode-based neuroimaging tools developed by the `Neural Systems and Behaviour Lab <https://www.monash.edu/medicine/psych/alex-fornito-lab>`_.

Features
--------
- Geometric eigenmode calculation from cortical surface meshes
- Eigendecomposition and reconstruction of cortical maps
- Neural field theory-based simulation of neural and BOLD activity
- Generative modeling of structural connectomes
- Incorporation of heterogeneity maps into all of the above
- Visualisation of cortical maps and matrices
- Triangular surface meshes for human, macaque, and marmoset cortices

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

Citing
------
J.C. Pang, K.M. Aquino, M. Oldehinkel, P.A. Robinson, B.D. Fulcher, M. Breakspear, A. Fornito, Geometric constraints on human brain function, Nature, 618, 566–574 (2023) (DOI: 10.1038/s41586-023-06098-1)

If you use the ``connectome`` module in your work, please cite:
F. Normand, M. Gajwani, T. Cao, J. Cruddas, A. Sangchooli, S. Oldham, A. Holmes, P.A. Robinson, J.C. Pang, A. Fornito, Geometric constraints on the architecture of mammalian cortical connectomes, BioRxiv (2025) (DOI: 10.1101/2025.09.17.676944)

License information
-------------------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (``cc-by-nc-sa``). See the `LICENSE <LICENCE-CC-BY-NC-SA-4.0.md>`_ file for details.
