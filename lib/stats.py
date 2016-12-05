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


def one_tailed_pvalue( dist, verbose=False ):
	if numpy.median(dist) < 0.0:
		TS = numpy.mean( -dist ) / numpy.std( -dist )
	else:
		TS = numpy.mean( dist ) / numpy.std( dist )
	pval = scipy.stats.norm.cdf( -TS )
	if verbose:
		print('p-val zed: %g' % pval)
	return pval


def freq_confidence_interval( dist, logs='10^', frac=0.95, verbose=False ):
	rem = (1.0-frac)/2.0*100.0
	freq_pctiles = tuple(numpy.percentile(dist, [50.0, rem, 100.0-rem]))
	if verbose:
		print(' '+logs+'%g [%g -- %g] CI' % freq_pctiles)
	return freq_pctiles


def compute_pctiles( dist, logs='10^', frac=0.95, bayes=True, verbose=False ):
	if bayes:
		return bayes_credible_region(dist, logs=logs, frac=frac, verbose=verbose)
	else:
		return freq_confidence_interval(dist, logs=logs, frac=frac, verbose=verbose)


def compute_pvalue( dist, bayes=True, verbose=False ):
	if numpy.median(dist) < 0.0:
		dist = -1.0*dist
	if bayes:
		return bayes_diff_pvalue( dist, verbose )
	else:
		return one_tailed_pvalue( dist, verbose )


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


def bayes_credible_region( dist, logs='10^', frac=0.95, verbose=False ):
	"""Compute the mode and frac% (95%) credible region"""
	dist = numpy.sort( dist )
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
	mode = scipy.optimize.brute( lambda x:-pdfkde(x), [[lb, ub]] )[0]
	bayes_pctiles = (mode, lb, ub)
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
	if verbose:
		print(' '+logs+'%g [%g -- %g] CR' % bayes_pctiles)
	return bayes_pctiles


def bayes_diff_pvalue( dist, verbose=False ):
	if numpy.median(dist) < 0.0:
		pval = (dist > 0.0).mean()
	else:
		pval = (dist < 0.0).mean()
	if verbose:
		print( 'p-val bayes: %g' % pval )
	return pval


##############################################################################
# This function determines either the
# - mode and confidence region (CR) - bayes=True (default)
# - median and confidence interval (CI) - bayes=False
# and returns a dictionary where the keys are the parameters and the
# values are tuples with (mode,lower-bound,upper-bound).
def chains_params( chainfiles, bayes=True, parlist=None, linpars=None, nburn=0 ):

	# Check if you've got one or more chains
	if isinstance(chainfiles, str):
		chainfiles = [chainfiles]

	pardics = []
	# Now evaluate each chain individually
	for chainfile in chainfiles:
		pdic,attrs = phymcmc.mcmc.load_mcmc_chain(chainfile,nburn)

		# Establish a parlist
		if parlist is None:
			parlist = list(attrs['parfit'])
			for key,value in pdic.items():
				try:
					len(value)
				except TypeError:
					continue
				if key not in parlist:
					parlist.append( key )
		if linpars is None:
			linpars = attrs['linpars']

		# Establish mode/median and CR/CI
		pardic = dict()
		pardic['linpars'] = linpars
		for key in parlist:
			# Copy distribution and use log if appropriate
			dis = 1.0*pdic[key]
			if key not in linpars:
				dis = numpy.log10(dis)
			# Simple report on individual chain (no comparison)
			pardic[key] = compute_pctiles(dis,bayes)
		pardics.append(pardic.copy())

	# Now return list of dicts
	return pardics


##############################################################################
# This function compares a list of chains, pairwise, and returns
# a dictionary where the keys are the parameters and the
# values are lists of the p-values for each pairwise comparison.
def chains_compare( chainfiles, bayes=True, parlist=None, linpars=None, nburn=0 ):

	# Do the combinatorics for pairwise-comparison
	nchains = len(chainfiles)
	pvaldic = dict()
	if parlist is not None:
		for key in parlist:
			pvaldic[key] = []
	for id1 in range(nchains-1):
		pdic1, attrs1 = phymcmc.mcmc.load_mcmc_chain(chainfiles[id1], nburn=nburn)

		# Build parlist if None
		if parlist is None:
			parlist = []
			for key,value in pdic1.items():
				try:
					len(value)
				except TypeError:
					continue
				if key not in parlist:
					parlist.append( key )
					pvaldic[key] = []

		if linpars is None:
			linpars = attrs1['linpars']

		# Now let's look at that second chain
		for id2 in range(id1+1,nchains):
			pdic2, attrs2 = phymcmc.mcmc.load_mcmc_chain(chainfiles[id2], nburn=nburn)
			nvals = min(len(pdic1['ssr']), len(pdic2['ssr']))

			# Compute p-value for each key
			for key in parlist:
				if key in linpars:
					dis1 = 1.0*pdic1[key]
					dis2 = 1.0*pdic2[key]
				else:
					dis1 = numpy.log10(pdic1[key])
					dis2 = numpy.log10(pdic2[key])
				# numpy.random.choice picks at random w replacement
				disdif = numpy.random.choice(dis1,5*nvals)-numpy.random.choice(dis2,5*nvals)
				# Compute p-value (to compare signif of difference)
				pval = compute_pvalue(disdif,bayes=bayes)
				pvaldic[key].append(pval)
	return pvaldic


def table_params( dic, parlist=None, parlabels=None, linpars=None ):

	analysis_linpars = dic[0]['linpars']
	if linpars is None:
		linpars = analysis_linpars
	if parlist is None:
		parlist = dic[0].keys()
		parlist.remove('linpars')
	if parlabels is None:
		parlabels = dict((key, key) for key in parlist)

	# Make table header
	nstrains = len(dic)
	table = '\hspace{-7em}%%\n\\begin{tabular}{l'
	table += 'c'*nstrains + '}\n'
	table += 'Parameter ' + ('& %d '*nstrains) % tuple(range(nstrains))
	table += '\\\\\n'
	# Make table: row=param, col=strain
	for key,label in zip(parlist,parlabels):
		table += '%s ' % label
		for strain in range(nstrains):
			if key in analysis_linpars:
				table += '& $%.3g\\ [%.2g,%.2g]$ ' % dic[strain][key]
			elif key in linpars:
				table += '& $%.3g\\ [%.2g,%.2g]$ ' % tuple([10.0**i for i in dic[strain][key]])
			else:
				table += '& $10^{%.3g\\ [%.2g,%.2g]}$ ' % dic[strain][key]
		table += '\\\\\n' # new line, new param
	# Make table footer
	table += '\\end{tabular}\n'
	return table


def table_compare( dic, labels=None, parlist=None, parlabels=None ):
	import itertools

	if parlist is None:
		parlist = dic.keys()
	if parlabels is None:
		parlabels = dict((key, key) for key in parlist)

	if labels is None:
		N = 1; dN = 2
		while N+1 < len(dic[parlist[0]]):
			N += dN; dN += 1
		labels = range(dN)
	labels = tuple('%s:%s'%label for label in itertools.combinations(labels,2))

	# Make table header
	nstrains = len(labels)
	table = '\hspace{-7em}%%\n\\begin{tabular}{l'
	table += 'c'*nstrains + '}\n'
	table += 'Parameter ' + ('& %s '*nstrains) % labels
	table += '\\\\\n'
	# Make table: row=param, col=strain-pair
	for key,label in zip(parlist,parlabels):
		table += '%s ' % label
		for idx,label in enumerate(labels):
			pval = dic[key][idx]
			if pval < 0.001:
				table += '& $\\mathbf{< 0.001}$ '
			elif pval < 0.05:
				table += '& \\textbf{%.3f} ' % pval
			else:
				table += '& %.3f ' % pval
		table += '\\\\\n' # new line, new param
	# Make table footer
	table += '\\end{tabular}\n'
	return table


def print_chain_parstats( chainfile1, chainfile2=None, parlist=None ):
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

		# Copy dist and use log if appropriate
		if key in attrs1['linpars']:
			dis1 = 1.0*pdic1[key]
			logs=''
			print('%s' % key)
		else:
			dis1 = numpy.log10(pdic1[key])
			print('%s [log10 distributed]' % key)
			logs='10^'
		# Simple report on individual chains (no comparison)
		compute_pctiles( dis1, logs=logs, bayes=False, verbose=True )
		compute_pctiles( dis1, logs=logs, bayes=True, verbose=True )

		if chainfile2:
			# Check in case parlist is not be same for both chains...
			assert len(pdic2[key]), 'Error: key %s not in chain %s. Use parlist optional argument to fix this problem.' % (key,chainfile2)
			if key in attrs1['linpars']:
				dis2 = 1.0*pdic2[key]
				logs = ''
			else:
				dis2 = numpy.log10(pdic2[key])
				logs = '10^'
			compute_pctiles( dis2, logs=logs, bayes=False, verbose=True )
			compute_pctiles( dis2, logs=logs, bayes=True, verbose=True )

			# Now compare the 2 chains
			###################################
			# ROC curve (if interested)
			#sts = roc( dis1, dis2 )
			#numpy.savetxt('/tmp/%s-roc' % key, sts)
			# Compute diff of chains distribution
			assert ((key in attrs1['linpars']) == (key in attrs2['linpars'])), 'Error: Your 2 chains do not agree on whether param %s is linear.' % key
			disdif = numpy.random.choice(dis1,5*nvals)-numpy.random.choice(dis2,5*nvals)
			# Compute p-value (to compare signif of difference)
			one_tailed_pvalue( disdif, verbose=True )
			bayes_diff_pvalue( disdif, verbose=True )
		print('')
	print( 'Time taken: %g s' % (time.time()-tstart) )

