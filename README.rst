.. image:: https://github.com/izachp/nsbtools/raw/main/docs/_static/logo.png
   :alt: Design by Gabriella Chan

Eigenmode-based neuroimaging tools developed by the `Neural Systems and Behaviour Lab <https://www.monash.edu/medicine/psych/alex-fornito-lab>`_.

Features
--------
- Geometric eigenmode calculation from surface meshes
- Eigendecomposition and reconstruction of brain maps
- Neural field theory simulation of neural and BOLD activity
- Incorporation of spatial heterogeneity from brain maps into all of the above
- Visualization of brain maps and heat maps
- Triangular surface meshes for human, macaque, and marmoset cortices

All human cortical surface meshes provided were obtained from the `neuromaps <https://neuromaps-main.readthedocs.io/en/stable/index.html>`_ package, which also provides a diverse set of cortical maps that can be used for our `EigenSolver`'s `hetero` parameter.
To compare brain maps while accounting for spatial autocorrelation, we recommend using the `eigenstrapping <https://eigenstrapping.readthedocs.io/en/stable/index.html>`_ package, which implements a rigorous null model based on geometric eigenmodes.

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
If you use the ``eigen`` module in your work, please cite the following paper:
J.C. Pang, K.M. Aquino, M. Oldehinkel, P.A. Robinson, B.D. Fulcher, M. Breakspear, A. Fornito, Geometric constraints on human brain function, Nature, 618, 566–574 (2023) (DOI: 10.1038/s41586-023-06098-1)

License information
-------------------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (``cc-by-nc-sa``). See the `LICENSE <LICENCE-CC-BY-NC-SA-4.0.md>`_ file for details.
