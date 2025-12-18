.. image:: https://github.com/NSBLab/neuromodes/raw/main/docs/_static/logo.png
  :alt: Design by Gabriella Chan

**NOTE: neuromodes is currently under active development. Breaking changes to function naming and behaviour may occur prior to a stable release.**

Eigenmode-based neuroimaging tools developed by the `Neural Systems and Behaviour Lab <https://www.monash.edu/medicine/psych/alex-fornito-lab>`_. Documentation can be found `here <https://neuromodes.readthedocs.io/en/latest/>`_.

Features
--------
.. image:: https://github.com/NSBLab/neuromodes/raw/main/docs/_static/overview.png

- ``eigen``: Compute geometric eigenmodes from cortical surface meshes, optionally incorporating spatial heterogeneity
- ``basis``: Decompose and reconstruct cortical maps using the modes, or another basis set
- ``waves``: Simulate neural activity and BOLD signals using the wave propagation model from `Pang et al. (2023) <https://doi.org/10.1038/s41586-023-06098-1>`_
- ``connectome``: Simulate structural connectomes using the generative model from `Normand et al. (2025) <https://doi.org/10.1101/2025.09.17.676944>`_
- ``io``: Access triangular surface meshes for human, macaque, and marmoset cortices
- ``morphometry`` *(under development)*: Quantify multiscale anatomical differences using the method from `Cao et al. (2024) <https://doi.org/10.1002/hbm.26640>`_
- ``parcellation`` *(under development)*: Create parcellations with any number of regions using the method from `Pang et al. (2025) <https://doi.org/10.1101/2025.01.30.635820>`_

To compare cortical maps while accounting for spatial autocorrelation, we recommend using the `eigenstrapping <https://eigenstrapping.readthedocs.io/en/stable/index.html>`_ package, which uses the geometric eigenmodes for rigorous null modelling.

Installation
------------
``neuromodes`` works with Python 3.9+, and can be pip-installed into your environment via:

::

  pip install git+https://github.com/NSBLab/neuromodes.git

This will clone ``main``, our most stable branch. To try out any newer features under development, clone from our ``dev`` branch instead via:

::

  pip install git+https://github.com/NSBLab/neuromodes.git@dev.

Alternatively, ``neuromodes`` can be added as a dependency to your ``pyproject.toml`` with `UV <https://docs.astral.sh/uv/>`_ via:

::

  uv add git+https://github.com/NSBLab/neuromodes.git.
|
If you wish to run the tutorials, please instead install our extra ``tutorials`` dependencies via:

::

  pip install "neuromodes[tutorials] @ git+https://github.com/NSBLab/neuromodes.git"

or

::

  uv add "neuromodes[tutorials] @ git+https://github.com/NSBLab/neuromodes.git"
|
If you encounter any problems, please consider `opening an issue <https://github.com/NSBLab/neuromodes/issues>`_. Meanwhile, try switching to the exact environment used for development via UV:

::

  git clone https://github.com/NSBLab/neuromodes
  cd neuromodes
  uv venv --python 3.12.10
  uv sync --frozen
|
Tests can be run with ``pytest`` via:

::

  cd neuromodes
  pip install pytest
  pytest tests

or

::

  cd neuromodes
  uv sync --extra testing
  pytest tests

Citing
------
If you use ``neuromodes`` in your work, please cite the following three papers:

J.C. Pang, K.M. Aquino, M. Oldehinkel, P.A. Robinson, B.D. Fulcher, M. Breakspear, A. Fornito, Geometric constraints on human brain function, Nature, 618, 566–574 (2023) (DOI: 10.1038/s41586-023-06098-1)

M\. Reuter, F-E. Wolter, N. Peinecke, Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and solids, Computer-Aided Design, 38(4), 342-366 (2006). (DOI: 10.1016/j.cad.2005.10.011)

C\. Wachinger, P. Golland, W. Kremen, B. Fischl, M. Reuter, BrainPrint: a discriminative characterization of brain morphology, Neuroimage, 109, 232-248 (2015). (DOI: 10.1016/j.neuroimage.2015.01.032)

If you use the ``model_connectome`` function, please also cite:

F\. Normand, M. Gajwani, T. Cao, J. Cruddas, A. Sangchooli, S. Oldham, A. Holmes, P.A. Robinson, J.C. Pang, A. Fornito, Geometric constraints on the architecture of mammalian cortical connectomes, BioRxiv (2025) (DOI: 10.1101/2025.09.17.676944)

Citations for cortical surface meshes and maps can be found in `neuromodes/data/included_data.csv <https://github.com/NSBLab/neuromodes/blob/main/neuromodes/data/included_data.csv>`_

License information
-------------------
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (``cc-by-nc-sa``). See the `LICENSE <LICENCE-CC-BY-NC-SA-4.0.md>`_ file for details.
