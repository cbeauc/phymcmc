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


def ztest( dist, verbose=False ):
	TS = numpy.mean( dist ) / numpy.std( dist )
	pval1 = scipy.stats.norm.sf( TS )
	pval2 = scipy.stats.norm.sf( -TS )
	pfin = 1.0 - abs(pval1 - pval2)
	if verbose:
		print('p-val zed: %g (%g,%g)' % (pfin,pval1,pval2))
	return (pfin,pval1,pval2)


def compute_pctiles( dist, logs='10^', frac=0.95, verbose=False ):
	rem = (1.0-frac)/2.0*100.0
	freq_pctiles = numpy.percentile(dist, [50.0, rem, 100.0-rem])
	bayes_pctiles = bayes_credible_region(dist, frac=frac)
	if verbose:
		print(' '+logs+'%g [%g -- %g] CI' % tuple(freq_pctiles))
		print(' '+logs+'%g [%g -- %g] CR' % tuple(bayes_pctiles))
	return (bayes_pctiles, freq_pctiles)


def compute_pvalue( dist, bayes=True, verbose=False ):
	if bayes:
		pval,_ = bayes_diff_pvalue( dist, verbose )
	else:
		pval,_,_ = ztest( dist, verbose )
	return pval


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
	mode = scipy.optimize.brute( lambda x:-pdfkde(x), [[lb, ub]] )[0]
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


def bayes_diff_pvalue( dist, verbose=False ):
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
			# Sort dist and use log if appropriate
			dis = numpy.sort(1.0*pdic[key])
			if key not in linpars:
				dis = numpy.log10(dis)
			# Simple report on individual chain (no comparison)
			pardic[key] = compute_pctiles(dis)[not bayes]
		pardics.append(pardic.copy())

	# Now return dict or list of dicts
	if len(pardics) == 1:
		return pardics[0]
	else:
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
		pdic1, attrs1 = phymcmc.mcmc.load_mcmc_chain(chainfiles[id1], nburn=0)

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
			pdic2, attrs2 = phymcmc.mcmc.load_mcmc_chain(chainfiles[id2], nburn=0)
			nvals = min(len(pdic1['ssr']), len(pdic2['ssr']))

			# Compute p-value for each key
			for key in parlist:
				if key in linpars:
					dis1 = 1.0*pdic1[key]
					dis2 = 1.0*pdic2[key]
				else:
					dis1 = numpy.log10(pdic1[key])
					dis2 = numpy.log10(pdic2[key])
				disdif = numpy.random.choice(dis1,nvals)-numpy.random.choice(dis2,nvals)
				if numpy.median(disdif) < 0.0:
					disdif = numpy.sort( -disdif )
				else:
					disdif = numpy.sort( disdif )
				# Compute p-value (to compare signif of difference)
				pval = compute_pvalue(disdif,bayes=bayes)
				pvaldic[key].append(pval)
	return pvaldic


def table_params( dic, parlist=None ):

	linpars = dic[0]['linpars']
	if parlist is None:
		parlist = dic[0].keys()
		parlist.remove('linpars')

	# Make table header
	table = '\hspace{-7em}%%\n\\begin{tabular}{'
	table += 'c' * (len(parlist)+1) + '}\n\n'
	table += repr(parlist).replace('\'','').replace(',',' &').replace('[','strain & ').replace(']',' \\\\\n')
	# Make table: row=strain, col=param
	for strain in range(len(dic)):
		table += '%d ' % strain
		for key in parlist:
			if key in linpars:
				table += '& $%.3g\\ [%.2g,%.2g]$ ' % dic[strain][key]
			else:
				table += '& $10^{%.3g\\ [%.2g,%.2g]}$ ' % dic[strain][key]
		table += '\\\\\n' # new line, new strain
	# Make table footer
	table += '\\end{tabular}\n'
	return table


def table_compare( dic, labels=None, parlist=None ):
	import itertools

	if parlist is None:
		parlist = dic.keys()

	if labels is None:
		N = 1; dN = 2
		while N+1 < len(dic[parlist[0]]):
			N += dN; dN += 1
		labels = range(dN)

	# Make table header
	table = '\hspace{-7em}%%\n\\begin{tabular}{'
	table += 'c' * (len(parlist)+1) + '}\n\n'
	table += repr(parlist).replace('\'','').replace(',',' &').replace('[','compare & ').replace(']',' \\\\\n')
	# Make table: row=strain, col=param
	for idx,label in enumerate(itertools.combinations(labels,2)):
		table += '%s:%s ' % label
		for key in parlist:
			pval = dic[key][idx]
			if pval < 0.0:
				table += '& \\textbf{< 0.001} '
			elif pval < 0.05:
				table += '& \\textbf{%.3f} ' % pval
			else:
				table += '& %.3f ' % pval
		table += '\\\\\n' # new line, new strain-pair
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
		compute_pctiles( dis1, logs=logs, verbose=True )

		if chainfile2:
			# Check in case parlist is not be same for both chains...
			assert len(pdic2[key]), 'Error: key %s not in chain %s. Use parlist optional argument to fix this problem.' % (key,chainfile2)
			if key in attrs1['linpars']:
				dis2 = numpy.sort(1.0*pdic2[key][-nvals:])
				logs = ''
			else:
				dis2 = numpy.sort(numpy.log10(pdic2[key][-nvals:]))
				logs = '10^'
			compute_pctiles( dis2, logs=logs, verbose=True )

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
			ztest( disdif, verbose=True )
			bayes_diff_pvalue( disdif, verbose=True )
		print('')
	print( 'Time taken: %g s' % (time.time()-tstart) )

