# Copyright (C) 2014 Catherine Beauchemin <cbeau@users.sourceforge.net>
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

import numpy
import scipy.stats
import scipy.interpolate
import phymcmc.mcmc

#
# =============================================================================
#
#                                   Utilities
#
# =============================================================================
#


def ztest( dist ):
	TS = numpy.mean( dist ) / numpy.std( dist )
	pval1 = scipy.stats.norm.sf( TS )
	pval2 = scipy.stats.norm.sf( -TS )
	pfin = 1.0 - abs(pval1 - pval2)
	print('p-val zed: %g (%g,%g)' % (pfin,pval1,pval2))


def table_print( dist, logs='10^', frac=0.95 ):
	rem = (1.0-frac)/2.0*100.0
	freq_pctiles = numpy.percentile(dist, [50.0, rem, 100.0-rem])
	bayes_pctiles = bayes_credible_region(dist, frac=frac)
	print(' '+logs+'%g [%g--%g] CI' % tuple(freq_pctiles))
	print(' '+logs+'%g [%g--%g] CR' % tuple(bayes_pctiles))
	return (bayes_pctiles, freq_pctiles)


def roc( dis1, dis2 ):
	xmin = min(min(dis1),min(dis2))
	xmax = max(max(dis1),max(dis2))
	xs = numpy.linspace(xmin,xmax,200)[::-1]
	sts = []
	for x in xs:
		fp = (dis1 > x).mean()
		tp = (dis2 > x).mean()
		sts.append([fp,tp])
	sts = numpy.array(sts)
	# Make sure you've got it right-way around
	if dis1.mean() > dis2.mean():
		sts = sts[:,::-1]
	rocauc = numpy.trapz(sts[:,1],sts[:,0])
	print('ROC-AUC: %g (p-val %g)' % (rocauc,1-rocauc) )
	print('ROC-Youden-J: %g' % max(abs(sts[:,1]-sts[:,0])) )
	return sts


def bayes_credible_region( dist, frac=0.95 ):
	"""Compute the mode and frac% (95%) credible region"""
	alpha = 1.-frac
	N = len(dist)
	# Get the factor% (def=95%) credible region
	ppf = scipy.interpolate.interp1d( numpy.arange(1.0,N+1)/N, dist, bounds_error=False, fill_value=0.0 )
	dx = 0.00001
	xv = numpy.arange(dx,alpha,dx)
	idx = numpy.argmin(ppf(frac+xv)-ppf(xv))
	lb,ub = ppf([xv[idx],frac+xv[idx]])
	# Get the mode
	pdfkde = scipy.stats.gaussian_kde( dist )
	mode = scipy.optimize.brute( lambda x:-pdfkde(x), [[lb, ub]] )
	# Plot to show me you got it right
	if False:
		import pylab
		pylab.hist( dist, 300 )
		xrng = numpy.linspace(dist.min(), dist.max(), 200)
		pylab.plot( xrng, 4000.0*pdfkde(xrng), 'r-')
		pylab.axvline( ppf(xv[idx]) )
		pylab.axvline( ppf(frac+xv[idx]) )
		pylab.axvline( mode, color='red' )
		pylab.show()
	return ( mode, lb, ub )


def bayes_diff_pvalue( dist ):
	# Get the mode
	qqart = (dist < 0.0).mean()
	if qqart < 5.0e-4:
		print('p-val bayes: < 0.001')
		return None
	pdfkde = scipy.stats.gaussian_kde(dist)
	xmx = scipy.optimize.newton(lambda x:pdfkde(x)-pdfkde(0.0), 2.0*numpy.median(dist))
	print (pdfkde(xmx), pdfkde(0.0), dist.mean())
	assert xmx > 0.0, 'Error: xmax (%g) found < 0.0' % xmx
	pval = 1.0-pdfkde.integrate_box_1d(0.0, xmx)
	if False:
		import pylab
		xrng = numpy.linspace( dist[0], dist[-1], 200 )
		pylab.plot( xrng, pdfkde(xrng), 'k-' )
		pylab.axvline( 0.0, color='red' )
		pylab.axvline( xmx, color='red' )
		pylab.show()
	print( 'p-val bayes: %g' % pval )


def compare_chains( chainfile1, chainfile2=None, parlist=None ):
	import time
	tstart = time.time()
	# FIXME: nburn?
	pdic1, attrs1 = phymcmc.mcmc.load_mcmc_chain(chainfile1, nburn=0)
	if parlist is None:
		parlist = list(attrs1['parfit'])
		for key,value in pdic1.items():
			try:
				len(value)
			except TypeError:
				continue
			if key not in parlist:
				parlist.append( key )

	if chainfile2 is None:
		attrs2 = attrs1
		nvals = len(pdic1['ssr'])
	else:
		pdic2, attrs2 = phymcmc.mcmc.load_mcmc_chain(chainfile2, nburn=0)
		nvals = min(len(pdic1['ssr']), len(pdic2['ssr']))

	for key in parlist:

		# Sort dist and use log if appropriate
		if key in attrs1['linpars']:
			dis1 = numpy.sort(1.0*pdic1[key][-nvals:])
			logs=''
			print('%s' % key)
		else:
			dis1 = numpy.sort(numpy.log10(pdic1[key][-nvals:]))
			print('%s [log10 distributed]' % key)
			logs='10^'
		# Simple report on individual chains (no comparison)
		table_print( dis1, logs=logs )

		if chainfile2:
			# Check in case parlist is not be same for both chains...
			assert len(pdic2[key]), 'Error: key %s not in chain %s. Use parlist optional argument to fix this problem.' % (key,chainfile2)
			if key in attrs1['linpars']:
				dis2 = numpy.sort(1.0*pdic2[key][-nvals:])
				logs = ''
			else:
				dis2 = numpy.sort(numpy.log10(pdic2[key][-nvals:]))
				logs = '10^'
			table_print( dis2, logs=logs )

			# Now compare the 2 chains
			###################################
			# ROC curve (if interested)
			#sts = roc( dis1, dis2 )
			#numpy.savetxt('/tmp/%s-roc' % key, sts)
			# Compute diff of chains distribution
			assert ((key in attrs1['linpars']) == (key in attrs2['linpars'])), 'Error: Your 2 chains do not agree on whether param %s is linear.' % key
			disdif = (numpy.random.choice(dis1,nvals)-numpy.random.choice(dis2,nvals))[:nvals]
			if numpy.median(disdif) < 0.0:
				disdif = numpy.sort( -disdif )
			else:
				disdif = numpy.sort( disdif )
			# Compute p-value (to compare signif of difference)
			ztest( disdif )
			bayes_diff_pvalue( disdif )
		print('')
	print( 'Time taken: %g s' % (time.time()-tstart) )

