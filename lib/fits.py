# Copyright (C) 2014-2021 Catherine Beauchemin <cbeau@users.sourceforge.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

from __future__ import print_function
import math
import numpy
import scipy.optimize
import sys

#
# =============================================================================
#
#                                   Utilities
#
# =============================================================================
#


def scost(lnpvec, model):
	if numpy.max(lnpvec) > 700.0:
		return float('inf')
	try:
		neglnprob = -model.get_lnprob(numpy.exp(lnpvec))
	except (ArithmeticError,ValueError) as e:
		print('** WARNING! Your code believes the parameters are valid but the call to ln_probability failed. Don\'t ignore this. Figure out why and fix this problem.', file=sys.stderr)
		print('pdic = '+repr(model.params.pardict), file=sys.stderr)
		print(e, file=sys.stderr)
		return float('inf')
	return neglnprob


def perform_fit(model, verbose=True, nreps=3):
	if verbose:
		print('#'+repr(model.params.parfit))
		print('#'+repr(model.params.vector))

	# It's best to fit parameters in log space
	lnpvec = numpy.log(model.params.vector)
	if verbose:
		lnprob = -scost(lnpvec, model)
		print( '# Starting lnprob = %g (to be maximized)\n' % lnprob )

	# Do nreps fits with rotating methods
	for rep in range(nreps):
		for meth in ['CG','TNC','COBYLA']:
			try:
				lsout = scipy.optimize.minimize(scost, lnpvec, args=(model), method=meth)
			except:
				continue
			if lsout.fun == float('inf'):
				continue
			lnprob = -lsout.fun
			lnpvec = lsout.x
			model.params.vector = numpy.exp(lnpvec)
			if verbose:
				print('# Rep=%d/%d using %s. success? %s (lnprob = %g)'%(rep+1,nreps,meth,lsout.success,lnprob))
				print( 'pdic = %s' % repr(model.params.pardict) )

	if verbose:
		print( '\n# Final values (lnprob = %g)' % lnprob )
		print( 'pdic = %s\n' % repr(model.params.pardict) )

	# Returns (best-fit parameters, lnprob)
	return (model.params, lnprob)


def linregress(x, y=None):
	"""
	Calculate a regression line

	This computes a least-squares regression for two sets of measurements.

	Parameters
	----------
	x, y : array_like
		two sets of measurements.  Both arrays should have the same length.
		If only x is given (and y=None), then it must be a two-dimensional
		array where one dimension has length 2.  The two sets of measurements
		are then found by splitting the array along the length-2 dimension.

	Returns
	-------
	slope : float
		slope of the regression line
	intercept : float
		intercept of the regression line
	full : dictionary
		Contains
			r2: R-squared
			dslope: 1-sigma error on slope
			dintercept: 1-sigma error on y-intercept

	"""
	TINY = 1.0e-20
	if y is None:  # x is a (2, N) or (N, 2) shaped array_like
		x = numpy.asarray(x)
		if x.shape[0] == 2:
			x, y = x
		elif x.shape[1] == 2:
			x, y = x.T
		else:
			msg = ("If only `x` is given as input, it has to be of shape "
					"(2, N) or (N, 2), provided shape was %s" % str(x.shape))
			raise ValueError(msg)
	else:
		x = numpy.asarray(x)
		y = numpy.asarray(y)
	n = len(x)
	xmean = numpy.mean(x, None)
	ymean = numpy.mean(y, None)

	# average sum of squares:
	ssxm, ssxym, ssyxm, ssym = numpy.cov(x, y, bias=1).flat
	r_num = ssxym
	r_den = numpy.sqrt(ssxm * ssym)
	if r_den == 0.0:
		r = 0.0
	else:
		r = r_num / r_den
		# test for numerical error propagation
		if r > 1.0:
			r = 1.0
		elif r < -1.0:
			r = -1.0

	df = n - 2
	t = r * numpy.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
	#prob = 2 * distributions.t.sf(numpy.abs(t), df)
	slope = ssxym / ssxm
	intercept = ymean - slope*xmean
	full = {}
	full['r2'] = r**2.0
	full['dslope'] = numpy.sqrt((1 - r**2) * ssym / ssxm / df)
	full['dintercept'] = full['dslope']*numpy.sqrt(numpy.sum(x**2.0)/n)

	return slope, intercept, full

