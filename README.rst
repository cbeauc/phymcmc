phymcmc
=======

**Wrapper of github.com/dfm/emcee providing convenient functions and scripts**

.. image:: https://img.shields.io/badge/GitHub-cbeauc%2Fphymcmc-blue.svg?style=flat
    :target: https://github.com/cbeauc/phymcmc
.. image:: https://img.shields.io/badge/license-GPL-blue.svg?style=flat
    :target: https://github.com/cbeauc/phymcmc/blob/master/LICENSE


phymcmc is a wrap of `github/dfm/emcee <https://github.com/dfm/emcee>`_.
It provides a class for describing one's model and another class for describing the parameters associated with the model. Having developed a model instance and a parameter instance of these classes, phymcmc functions, scripts and modules can be used to, for example, fit the model to data to obtain so-called best-fit parameters; or obtain posterior parameter likelihood distributions (PostPLDs) based on the MCMC process implemented in `emcee <https://github.com/dfm/emcee>`_, save the results in a phymcmc-defined hdf5 file format. The phymcmc hdf5 files can be parsed/analysed by a number of other phymcmc-provided scripts to, for example, draw diagnostic plots to evaluate whether the runs have converged, draw histograms of individual parameter's PostPLDs, draw pair-wise PostPLDs to identify parameter correlations using `github/dfm/corner.py <https://github.com/dfm/corner.py>_`, or obtain p-values when comparing parameter PostPLDs for 2 different sets of data (e.g., analysis of a wild-type versus a mutant virus). It has been used in a number of publications by the `phymbie <https://phymbie.physics.ryerson.ca>`_ research group (e.g. `

Attribution
-----------

If you make use of this code, make sure to cite it.

The BibTeX entry for the paper is::

	@MANUAL{phymcmc,
		AUTHOR = "Catherine A. A. Beauchemin",
		TITLE = "{phymcmc}: {A} convenient wrapper for emcee",
		YEAR = "2019",
		PUBLISHER = "{GitHub}",
		JOURNAL = "{GitHub} repository",
		HOWPUBLISHED = "\url{https://github.com/cbeauc/phymcmc}"
	}


License
-------

Copyright 2014-2019 Catherine Beauchemin and contributors.

phymcmc is free software made available under the GNU General Public License Version 3. For details see the LICENSE file.
