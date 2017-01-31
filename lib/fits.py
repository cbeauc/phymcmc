# Copyright (C) 2014-2016 Catherine Beauchemin <cbeau@users.sourceforge.net>
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
import numpy
import scipy.optimize
PosInf = float('+inf')

#
# =============================================================================
#
#                                   Utilities
#
# =============================================================================
#


def rcost(pvec, model, maxssr):
	return numpy.ones(pvec.shape)*scost(pvec, model, maxssr)

def scost(pvec, model, maxssr):
	pvec = 10.0**pvec
	try:
		nssr = model.get_normalized_ssr(pvec)
	except ValueError: # Invalid parameters
		return maxssr
	except: # Unknown error
		print('WARNING: Your code believes the parameters are valid but the call to get_normalized_ssr failed. Don\'t ignore this. Figure out why and fix this problem.')
		print('params:', model.params.pardict)
		return maxssr
	import math
	if math.isnan( nssr ):
		print('WARNING: lnprob encountered NaN SSR (and returned -inf) for:\n '+repr(model.params.pardict), file=sys.stderr)
		return maxssr
	return nssr


def perform_fit(model, verbose=True, maxssr=PosInf, rep_fit=3):
	if verbose:
		print(model.params.parfit)
		print(model.params.vector)

	# It's best to fit parameters in log space
	pvec = numpy.log10(model.params.vector)
	if verbose:
		ssr = scost(pvec, model, maxssr)
		print( 'Starting ssr = %g\n' % ssr )

	# Do rep_fit fits w leastsq, a wrapper of MINPACK's Levenberg-Marquardt
	for rep in range(rep_fit):
		lsout = scipy.optimize.leastsq(rcost, pvec, args=(model,maxssr), maxfev=7200, full_output=True)[0:3]
		pvec = lsout[0]
		model.params.vector = 10.0**pvec
		ssr = lsout[2]['fvec'][0]
		if verbose:
			print( 'Levenberg-Marquardt, rep %d (ssr = %g)' % (rep,ssr) )
			print( model.params.pardict )

	# One long but more accurate fit using the Nelder-Mead downhill simplex
	[pvec,ssr] = scipy.optimize.fmin(scost, pvec, args=(model,maxssr), full_output=True, disp=False)[0:2]
	model.params.vector = 10.0**pvec
	if verbose:
		print( 'Nelder-Mead (ssr = %g)' % ssr )
		print( model.params.pardict )

	# One last fit w leastsq, a wrapper of MINPACK's Levenberg-Marquardt
	lsout = scipy.optimize.leastsq(rcost, pvec, args=(model,maxssr), maxfev=7200, full_output=True)
	pvec = lsout[0]
	model.params.vector = 10.0**pvec
	ssr = lsout[2]['fvec'][0]
	if verbose:
		print( 'Levenberg-Marquardt (final fit, ssr = %g)' % ssr )
		print( model.params.pardict )

	# Returns (best-fit parameters, SSR)
	return (model.params, ssr)


def mock_yield_coeff(data):
	"""Pre-computes part of Mock-Yield SSR calculation.

	argument: data a 2-column array with (time,virus)
	returns: a tuple (arg1,arg2) from which you can compute
	      SSR = arg1 + arg2 * clear
	where 'clear' is your clearance rate.
	"""
	import math
	td = data[:,0].mean()-data[:,0]
	vd = numpy.log10(data[:,1]).mean()-numpy.log10(data[:,1])
	return (vd, math.log10(math.e)*td)


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

