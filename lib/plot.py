# Copyright (C) 2014-2017 Catherine Beauchemin <cbeau@users.sourceforge.net>
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
### plotting STUFF
import matplotlib
import matplotlib.style
matplotlib.style.use('classic')
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
params = {
	'xtick.labelsize': 14,
	'ytick.labelsize': 14,
	'axes.titlesize': 'medium',
	'axes.labelsize': 'medium',
	'legend.fontsize': 'medium',
	'font.family': 'serif',
	'font.serif': 'Computer Modern Roman',
	'font.size': 14,
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
	def __init__(self, (gridheight,gridwidth), hspace=0.35, wspace=0.2, rwidth=3.0, rheight=2.8):
		import matplotlib.pyplot
		self.gh = gridheight
		self.gw = gridwidth
		# Setup the figure looking nice
		self.fig = matplotlib.pyplot.figure()
		self.fig.set_size_inches(rwidth*self.gw,rheight*self.gh)
		matplotlib.pyplot.subplots_adjust(hspace=hspace, wspace=wspace)

	def subaxes(self, idx, *args, **kwargs):
		return matplotlib.pyplot.subplot2grid((self.gh,self.gw), (idx/self.gw,idx%self.gw), *args, **kwargs)


def triangle( parlist, rawlabels, chain_file, nburn=0, linpars=None, weights=None ):
	pardict, chainattrs = phymcmc.mcmc.load_mcmc_chain( chain_file, nburn=nburn )
	if linpars is None:
		linpars = chainattrs['linpars']
	labels = rawlabels[:]
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
		else:
			data.append( numpy.log10( pardict[p] ) )
			labels[i] = r'log$_{10}$ '+labels[i]
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


def choose_bins(x, nbins, linear=False):
	# FIXME: percentile will change based on weights but not accouted for!!!
	if linear:
		whm = numpy.percentile(x,[15.87,50,84.13])
		return numpy.linspace(5.0*whm[0]-4.0*whm[1],5.0*whm[2]-4.0*whm[1],nbins)
	whm = numpy.percentile(numpy.log10(x),[15.87,50,84.13])
	return 10.**numpy.linspace(5.0*whm[0]-4.0*whm[1],6.0*whm[1]-5.0*whm[0],nbins)


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
	b = numpy.hstack(zip(b,b))
	n = numpy.hstack([0.0]+zip(n,n)+[0.0])
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


def hist_grid( keys, chainfiles, colors, dims=None, labels=None, bins=50, relative=[], nburn=0, linpars=None, weights=None, scaling='normalized', hist=hist ):
	# Set the arrangement/dimensions of the hist grid
	if dims is None:
		gh = int(math.floor(math.sqrt(len(keys)/1.618)))
		gw = int(math.ceil(1.0*len(keys)/gh))
	else:
		(gh,gw) = dims
		assert len(keys) <= (gw*gh), "Grid dimensions %s cannot fit keys (%d)" % (repr(dims),len(keys))

	if labels is None:
		labels = keys

	# Setup the figure looking nice
	gridfig = grid_plot((gh,gw))

	# Load the parameters of each chain file
	pardicts = dict((key, []) for key in keys)
	bestfits = dict((key, []) for key in keys)
	clen = 1.0e30
	for i,cf in enumerate(chainfiles):
		pdic,chainattrs = phymcmc.mcmc.load_mcmc_chain( cf, nburn=nburn )
		if linpars is None:
			clinpars = chainattrs['linpars']
		else:
			clinpars = linpars
		clen = min( clen, len(pdic[keys[0]]) )
		for key in keys:
			pardicts[key].append( pdic[key] )
			bestfits[key].append( numpy.median( pdic[key] ) )

	# Now start plotting each histogram
	for i,key in enumerate(keys):
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
			n = hist(ax, x, bins=bins, linear=(key in clinpars), scaling=scaling, weights=weights, color=colors[cfn])
			nmax = max(n,nmax)

		ax.set_xlabel(labels[i])
		ax.set_ylim(0, 1.1*nmax)
	return gridfig
