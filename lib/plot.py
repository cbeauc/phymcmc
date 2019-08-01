# Copyright (C) 2014-2019 Catherine Beauchemin <cbeau@users.sourceforge.net>
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

import math
import numpy
import phymcmc.mcmc
## This imports the "old" autocorr.py from
##		a past version of Dan Foreman-Mackey's emcee
import phymcmc.autocorr as dfmautocorr

### plotting STUFF
import matplotlib
matplotlib.use('Agg')
params = {
	# Makes things better
	'xtick.labelsize': 14.0,
	'ytick.labelsize': 14.0,
	'axes.titlesize': 'medium',
	'axes.labelsize': 'medium',
	'legend.fontsize': 'medium',
	'font.family': 'serif',
	'font.serif': 'Computer Modern Roman',
	'font.size': 14.0,
	'text.usetex': True
}
matplotlib.rcParams.update(params)

#
# =============================================================================
#
#                                   Utilities
#
# =============================================================================
#


class grid_plot(object):
	def __init__(self, ghgw, hspace=0.35, wspace=0.2, rwidth=3.0, rheight=2.8):
		import matplotlib.pyplot
		self.gh = ghgw[0]
		self.gw = ghgw[1]
		# Setup the figure looking nice
		self.fig = matplotlib.pyplot.figure()
		self.fig.set_size_inches(rwidth*self.gw,rheight*self.gh)
		matplotlib.pyplot.subplots_adjust(hspace=hspace, wspace=wspace)

	def subaxes(self, idx, *args, **kwargs):
		return matplotlib.pyplot.subplot2grid((self.gh,self.gw), (idx//self.gw,idx%self.gw), *args, **kwargs)


def triangle( chain_file, parlist=None, labels=None, nburn=0, linpars=None, weights=None ):
	pardict, chainattrs = phymcmc.mcmc.load_mcmc_chain( chain_file, nburn=nburn )
	if parlist is None:
		parlist = chainattrs['parfit']
	if labels is None:
		labels = list(parlist)
	else:
		labels = labels[:]
	if linpars is None:
		linpars = chainattrs['linpars']
	# Data
	data = []
	# Best-fit values
	truths = [] # blue vertical line in hist plot (dashed-black is median)
	imaxlnprob = pardict['lnprob'].argmax() # index of max lnprob
	# Range for parameter (x) axis
	extents = []
	for i,p in enumerate(parlist):
		if p in linpars:
			data.append( pardict[p] )
			if '_' in labels[i]:
				labels[i] = r'\verb|%s|' % labels[i]
		else:
			data.append( numpy.log10( pardict[p] ) )
			if '_' in labels[i]:
				labels[i] = r'log$_{10}$ \verb|%s|' % labels[i]
			else:
				labels[i] = r'log$_{10}$ %s' % labels[i]
		truths.append( data[-1][imaxlnprob] )
	data = numpy.vstack( data ).T
	from .emcee import corner as dfmtriangle
	fig = dfmtriangle.corner(data, labels=labels, truths=truths, weights=weights)
	# Now add a histogram for lnprob
	x = pardict['lnprob']
	ax = fig.axes[len(parlist)-2]
	ax.set_visible(True)
	ax.set_frame_on(True)
	ax.set_title('lnProb')
	ax.set_yticklabels([])
	tbins = numpy.linspace(x.max()-3.5*(x.max()-numpy.median(x)),x.max(),50)
	ax.hist(x, bins=tbins, normed=True, color='black', histtype='step')
	return fig


def square( chainfile, parlist=None, labels=None, color='b', nbins=20, reset=True, gridfig=None, linpars=None, nburn=0 ):
	# Load chain
	pdic,chainattrs = phymcmc.mcmc.load_mcmc_chain( chainfile, nburn=nburn )
	if parlist is None:
		parlist = chainattrs['parfit']
	Npars = len(parlist)
	if reset:
		gridfig = grid_plot((Npars,Npars))
	else:
		assert gridfig is not None, "If reset is False, you must provide a gridfig."
	if labels is None:
		labels = parlist
	if linpars is None:
		linpars = chainattrs['linpars']
	# Deciding index binning
	splitidxs = numpy.array_split(numpy.arange(len(pdic['lnprob'])),nbins)
	pid = -1
	for xkey,xlab in zip(parlist,labels):
		sortidxs = numpy.argsort(pdic[xkey])
		xvals = pdic[xkey][sortidxs]
		likeli = pdic['lnprob'][sortidxs]
		for ykey,ylab in zip(parlist,labels):
			pid +=1
			if xkey == ykey:
				continue
			yvals = pdic[ykey][sortidxs]
			# Compile best-fit pair for each block of values
			coords = []
			for splitidx in splitidxs:
				bfidx = numpy.argmax( likeli[splitidx] )
				coords.append([xvals[splitidx][bfidx],yvals[splitidx][bfidx]])
			coords = numpy.array(coords)
			# Plot the coords
			if reset:
				ax = gridfig.subaxes(pid)
			else:
				ax = gridfig.fig.axes[pid]
			ax.plot(coords[:,0],coords[:,1],'o',color=color)
			if reset:
				ax.set_xlabel(xlab)
				ax.set_ylabel(ylab)
				if xkey not in linpars:
					ax.set_xscale('log')
				if ykey not in linpars:
					ax.set_yscale('log')
	return gridfig


def choose_bins(x, nbins, linear=False):
	# FIXME: percentile will change based on weights but not accouted for!!!
	if linear:
		whm = numpy.percentile(x,[15.87,50,84.13])
		return numpy.linspace(5.0*whm[0]-4.0*whm[1],5.0*whm[2]-4.0*whm[1],nbins)
	whm = numpy.percentile(numpy.log10(x),[15.87,50,84.13])
	return 10.**numpy.linspace(5.0*whm[0]-4.0*whm[1],6.0*whm[1]-5.0*whm[0],nbins)


def violinhist( ax, ysets, xlocs, ybins, edgecolor='tab:blue', facecolor='#a5c9e1', linewidth=0.0 ):
	for xloc,yset,ybin in zip(xlocs,ysets,ybins):
		# Select the binning
		if isinstance(ybin,int):
			tbins = choose_bins( yset, ybin, linear=linear )
		else:
			tbins = ybin
		# Distribute data into the bins
		n,b = numpy.histogram( yset, tbins )
		n = n/numpy.max(n) * numpy.diff(xlocs)[0]*0.45
		b = list(zip(b,b)); b = numpy.hstack(b+b[-1::-1])
		n = numpy.hstack([0.0]+list(zip(n,n))+[0.0])
		n = numpy.hstack((xloc-n,xloc+n[-1::-1]))
		ax.fill( n, b, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth )
	return ax


def hist( ax, x, bins, linear=False, scaling=None, weights=None, color='blue'):
	# Select the binning
	if isinstance(bins,int):
		tbins = choose_bins( x, bins, linear=linear )
	else:
		tbins = bins
	# Distribute data into the bins
	n,b = numpy.histogram( x, tbins, weights=weights )
	if scaling == 'density':
		n = n * 1.0 / numpy.sum(n) / numpy.diff(b)
	elif scaling == 'normalized':
		n = n * 1.0 / numpy.max(n)
	# Now we're ready to make a beautiful histogram
	facecol = list(matplotlib.colors.colorConverter.to_rgb(color))
	facecol = tuple( facecol + [0.2] ) # Add alpha for face
	b = numpy.hstack(list(zip(b,b)))
	n = numpy.hstack([0.0]+list(zip(n,n))+[0.0])
	ax.fill( b, n, facecolor=facecol, edgecolor=color, linewidth=2.0 )
	if not linear:
		ax.set_xscale('log')
	return n.max()


def lalhist( ax, x, bins, linear=False, scaling=None, weights=None, color='blue'):
	import lalrate
	# Select the binning
	assert isinstance(bins,int), 'bins argument should be an integer, not actual bin edges'
	if linear:
		bins = lalrate.NDBins((lalrate.LinearBins(x.min(),x.max(),5*bins),))
	else:
		bins = lalrate.NDBins((lalrate.LogarithmicBins(x.min(),x.max(),5*bins),))
	pdf = lalrate.BinnedArray(bins)
	# Distribute data into the bins
	pdf.array[:],__ = numpy.histogram( x, numpy.hstack((bins.lower()[0],bins.upper()[0][-1])), weights=weights )
	#window = lalrate.gaussian_window(5)
	window = numpy.arange(21, dtype = "double") - 10.
	window = numpy.exp(-.5 * window**2. / (10**2 / 3.))
	window /= window.sum()
	lalrate.filter_array(pdf.array, window)
	if scaling == 'density':
		pdf.to_pdf()
	# Now we're ready to make a beautiful histogram
	facecol = list(matplotlib.colors.colorConverter.to_rgb(color))
	facecol = tuple( facecol + [0.2] ) # Add alpha for face
	ax.fill( pdf.centres()[0], pdf.array, facecolor=facecol, edgecolor=color, linewidth=2.0 )
	if not linear:
		ax.set_xscale('log')
	return pdf.array.max()


def hist_grid( chainfiles, parlist=None, labels=None, colors=None, dims=None, bins=50, relative=[], nburn=0, linpars=None, weights=None, scaling='normalized', hist=hist ):

	# Load first chainfile to have a peak at content
	pdic,chainattrs = phymcmc.mcmc.load_mcmc_chain( chainfiles[0], nburn=nburn )
	# Decide on parlist if not given
	if parlist is None:
		parlist = chainattrs['parfit']
	# Set the arrangement/dimensions of the hist grid
	if dims is None:
		gh = int(math.floor(math.sqrt(len(parlist)/1.618)))
		gw = int(math.ceil(1.0*len(parlist)/gh))
	else:
		(gh,gw) = dims
		assert len(parlist) <= (gw*gh), "Grid dimensions %s cannot fit parlist (%d)" % (repr(dims),len(parlist))
	# Setup the figure looking nice
	gridfig = grid_plot((gh,gw))
	# Set labels
	if labels is None:
		labels = parlist
	# Set colour iteration
	if colors is None:
		colors = ['blue','red','green','black','gold'][:len(parlist)]

	# Load the parameters of each chain file
	pardicts = dict((key, []) for key in parlist)
	clen = 1.0e30
	for i,cf in enumerate(chainfiles):
		if i: # If this is not the first chainfile
			pdic,chainattrs = phymcmc.mcmc.load_mcmc_chain( cf, nburn=nburn )
		else: # If it is the first chainfile
			if linpars is None:
				linpars = chainattrs['linpars']
		clen = min( clen, len(pdic[parlist[0]]) )
		for key in parlist:
			if key in pdic.keys():
				pardicts[key].append( pdic[key] )
			else: # if key exists in one chain but not in another
				pardicts[key].append( 0.0 )

	# Now start plotting each histogram
	for i,key in enumerate(parlist):
		ax = gridfig.subaxes(i)
		nmax = 0.0
		for cfn in range(len(chainfiles)):
			if len(relative):
				x = pardicts[key][cfn]/numpy.median( pardicts[key][relative[cfn]] )
				#if cfn == relative[cfn]:
				#	x = pardicts[key][cfn]/numpy.median( pardicts[key][cfn] )
				#else:
				#	x = pardicts[key][cfn][-clen:]/pardicts[key][relative[cfn]][-clen:]
			else:
				x = pardicts[key][cfn]
			if type(x) is float: # if x is not an array, don't plot
				continue
			n = hist(ax, x, bins=bins, linear=(key in linpars), scaling=scaling, weights=weights, color=colors[cfn])
			nmax = max(n,nmax)

		ax.set_xlabel(labels[i])
		ax.set_ylim(0, 1.1*nmax)
	return gridfig


def sigma_plot(ax, x, data, pcents=[68.2689492137,95.4499736104], colors=('r',(0.8,0.8,0.8),(0.5,0.5,0.5))):
	# First, massage the data
	y = []
	for adat in data:
		y.append( numpy.interp(x,adat[:,0],adat[:,1]) )
	# Median, 68%, 95%
	percentiles = [50.0]
	for pcent in numpy.sort(pcents):
		tmp = (100.0-pcent)/2.0
		percentiles += [tmp, 100-tmp]
	percentiles = percentiles[-1::-1]
	# Prepare the contour lines
	y = numpy.percentile(numpy.vstack(y),percentiles,axis=0).T
	for idx,col in enumerate(colors[-1:0:-1]):
		ax.fill_between(x, y[:,2*idx],y[:,2*idx+1], facecolor=col,linewidth=0)
	ax.plot(x,y[:,-1],color=colors[0],linestyle='-')


def brooksgelman( par ):
	""" This function implements the Brooks-Gelman method for evaluating
		the convergence of an MCMC chain. Argument "par" is an array of
		dimensions (nrow,ncols) = (nsteps,nwalker) for one parameter. """

	# In the Brooks and Gelman paper, the variables are as follows
	# m = # chains or walkers
	# n = # of steps or iterations (after applying the burn-in)
	# psi_j,t = value of param at iteration t of n for chain/walker j of m
	bgstats = {}

	# Param array dimensions (n,m)
	(nsteps,nwalkers) = par.shape
	parcumsum = par.cumsum(axis=0)
	divi = numpy.resize(numpy.arange(nsteps)+1.0,(nwalkers,nsteps)).T

	# Between-chain variance
	# B/n = 1/(m-1) sum_{j=1}^m ( mean(psi_j,all-t) - mean(psi_all-j,all-t) )^2
	BoverN = numpy.var( parcumsum/divi, axis=1, ddof=1 )[1:]

	# Within-chain variance
	# W = 1/[m(n-1)] sum_{j=1}^m sum_{t=1}^n [ psi_j,t - mean(psi_j,all-t) ]^2
	bgstats['W'] = numpy.sum((numpy.cumsum(par**2.0,axis=0)-parcumsum**2.0/divi)[1:,:]/(divi[1:,:]-1.0),axis=1)/nwalkers

	# Pooled posterior variance estimate
	# Vhat = (n-1)/n W + B/n + B/(m*n) = (n-1)/n W + (1+1/m) B
	bgstats['Vhat'] = (divi[1:,0]-1.0)/divi[1:,0] * bgstats['W'] + (1.0+1.0/nwalkers) * BoverN

	# (over-)estimated variance ratio of pooled/within-chain inferences
	#	aka potential scale reduction factor (PSRF)
	#	Should ideally be close to one for convergence
	bgstats['Rhat'] = bgstats['Vhat']/bgstats['W']
	return bgstats


def complete_diagnostics_chart( gridfig, baseidx, key, pararray, lin=False ):
	i = 0
	N = len(pararray)
	iters = range(N)
	if lin:
		yscale = 'linear'
	else:
		yscale = 'log'

	# Percentiles
	ax = gridfig.subaxes(baseidx+i)
	i += 1
	ax.set_title(r'Median, 1$\sigma$, $2\sigma$')
	ax.set_ylabel(key.replace('_','\_'))
	ax.set_yscale( yscale )
	tmp = numpy.percentile(pararray,[50.0,15.87,84.13,2.275,97.72],axis=1)
	ax.plot(iters, tmp[0], '-', color='tab:orange', label='median')
	ax.plot(iters, tmp[1], '-', iters, tmp[2], '-', color='tab:blue', label=r'$1\sigma$')
	ax.plot(iters, tmp[3], 'k-', iters, tmp[4], 'k-', label=r'$2\sigma$')

	# Actual chain
	ax = gridfig.subaxes(baseidx+i)
	i += 1
	ax.plot(iters,pararray)
	ax.set_title('raw walks')
	ax.set_yscale( yscale )

	# Running mean
	ax = gridfig.subaxes(baseidx+i)
	i += 1
	ax.set_title('cumm mean')
	tmp = []
	if lin:
		for chn in pararray.T:
			tmp.append( numpy.cumsum(chn)/(numpy.arange(N)+1) )
	else:
		for chn in pararray.T:
			tmp.append( 10.0**(numpy.cumsum(numpy.log10(chn))/(numpy.arange(N)+1)) )

	ax.plot(iters,numpy.array(tmp).T)
	ax.set_yscale( yscale )

	# Integrated autocorrelation time since start
	#	computed from average pos of all walkers at each step.
	#	emcee developer suggests a burn-in time of ~ 10x autocor time.
	ax = gridfig.subaxes(baseidx+i)
	i += 1
	ax.set_title(r'Integrated autocorr time')
	pmean = pararray.mean(axis=1)
	idxs = len(pmean)//200
	idxs = list(range(idxs,len(pmean),idxs))+[len(pmean)-1]
	acorr = []
	for idx in idxs:
		acorr.append( dfmautocorr.integrated_time(pmean[:idx]) )
	ax.plot(idxs,acorr)

	# Brooks-Gelman
	dis = 1
	bgstats = brooksgelman( pararray[dis-1:,:] )
	title = 'Brooks-Gelman'
	# W along Vhat
	ax = gridfig.subaxes(baseidx+i)
	i += 1
	ax.plot(iters[dis:],numpy.sqrt(bgstats['W']),label='W')
	ax.plot(iters[dis:],numpy.sqrt(bgstats['Vhat']),label=r'$\hat{V}$')
	ax.set_xlim(iters[dis],iters[-1])
	ax.legend(loc='best')
	ax.set_title( title )
	# Rhat (uncorrected)
	ax = gridfig.subaxes(baseidx+i)
	i += 1
	ax.plot(iters[dis:],numpy.sqrt(bgstats['Rhat']),label=r'$\hat{R}$')
	ax.axhline(1.0,color='k',linestyle='-')
	ax.set_ylim(0.9,max(numpy.sqrt(bgstats['Rhat'][len(bgstats['W'])//4]),1.5))
	ax.set_xlim(iters[dis],iters[-1])
	ax.legend(loc='best')
	ax.set_title( title )


def diagnostics( chain_file, savefile, nburn=0, exclude_lnprob=False, parlist=None ):
	# Get the chain first
	pardict, chainattrs = phymcmc.mcmc.load_mcmc_chain(chain_file, nburn=nburn)

	# Construct parlist from chain if none provided
	if parlist is None:
		parlist = chainattrs['parfit']
		if exclude_lnprob:
			parlist = parlist[1:]

	# Check if reshaping the parameter arrays will restitute the chains
	# correctly (rather than accidentally spread chains across columns).
	remainder = chainattrs['filledlength'] % chainattrs['nwalkers']
	assert remainder == 0.0, 'Error: your chain could not be reshaped into (nsteps,nwalkers). There are %d samples too many.' % remainder

	# Now get ready for the plot of all plots
	ndiags = 6
	gridfig = grid_plot((len(parlist)+1,ndiags))

	# Construct the (nsteps,nwalkers) array for each parameter
	# and run the requested conversion test method.
	for idx,key in enumerate(parlist):
		pararray = pardict[key].reshape((-1,chainattrs['nwalkers']))
		complete_diagnostics_chart( gridfig, idx*ndiags, key, pararray, lin=(key in chainattrs['linpars']) )

	# Add more overall (not per-parameter) diagnostics (?)
	# Acceptance fraction
	ax = gridfig.subaxes(len(parlist)*ndiags)
	ax.plot(range(len(pararray)-1),chainattrs['acceptance_fraction'])
	ax.set_title(r'acceptance fraction')

	# Saving the figure
	import tempfile
	import subprocess
	_,tmpfname = tempfile.mkstemp(suffix='.png')
	gridfig.fig.savefig(tmpfname, bbox_inches='tight')
	subprocess.call('convert %s %s_diag.pdf'% (tmpfname,savefile), shell=True)
	subprocess.call('rm -f %s'% tmpfname, shell=True)
