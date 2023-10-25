phymcmc
=======

**Wrapper of github.com/dfm/emcee providing convenient functions and scripts**

.. image:: https://img.shields.io/badge/GitHub-cbeauc%2Fphymcmc-blue.svg?style=flat
    :target: https://github.com/cbeauc/phymcmc
.. image:: https://img.shields.io/badge/license-GPL-blue.svg?style=flat
    :target: https://github.com/cbeauc/phymcmc/blob/master/LICENSE


phymcmc is a wrap of `github/dfm/emcee <https://github.com/dfm/emcee>`_.
It provides a model class a parameter class to describe one's model and associated parameters. Having defined a model and a parameter instance, phymcmc functions, scripts and modules can be used to, e.g., fit the model to data to obtain so-called best-fit parameters; or obtain posterior parameter likelihood distributions (PostPLDs) based on the MCMC process implemented in `emcee <https://github.com/dfm/emcee>`_, save the results in a phymcmc-defined hdf5 file format. The phymcmc hdf5 files can be parsed/analysed by phymcmc-provided scripts to, e.g., draw diagnostic plots to evaluate whether the runs have converged, draw histograms of individual parameter's PostPLDs, draw pair-wise PostPLDs to identify correlations using `github/dfm/corner.py <https://github.com/dfm/corner.py>`_, or obtain p-values when comparing parameter PostPLDs for 2 different sets of data (2 hdf5 files). It has been used in a number of publications by the `phymbie <https://phymbie.physics.ryerson.ca/publications>`_ research group. See the examples directory to get started.

Attribution
-----------

If you make use of this code, make sure to cite it.

The BibTeX entry is::

	@MANUAL{phymcmc,
		AUTHOR = "Catherine A. A. Beauchemin",
		TITLE = "{phymcmc}: {A} convenient wrapper for emcee",
		YEAR = "2022",
		PUBLISHER = "{GitHub}",
		JOURNAL = "{GitHub} repository",
		HOWPUBLISHED = "\url{https://github.com/cbeauc/phymcmc}"
	}


License
-------

Copyright 2014-2022 Catherine Beauchemin and contributors.

phymcmc is free software made available under the GNU General Public License Version 3. For details see the LICENSE file.
