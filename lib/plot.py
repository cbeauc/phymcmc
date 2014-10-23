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

import math
import numpy
import phymcmc.mcmc
### plotting STUFF
import matplotlib
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
	def __init__(self, (gridheight,gridwidth), hspace=0.35, wspace=0.2):
		import matplotlib.pyplot
		self.gh = gridheight
		self.gw = gridwidth
		# Setup the figure looking nice
		self.fig = matplotlib.pyplot.figure()
		self.fig.set_size_inches(3*self.gw,2.8*self.gh)
		matplotlib.pyplot.subplots_adjust(hspace=hspace, wspace=wspace)

	def subaxes(self, idx):
		return matplotlib.pyplot.subplot2grid((self.gh,self.gw), (idx/self.gw,idx%self.gw))


def triangle( parlist, rawlabels, chain_file, nburn=0, linpars=None ):
	pardict, chainattrs = phymcmc.mcmc.load_mcmc_chain( chain_file, nburn=nburn )
	if linpars is None:
		linpars = chainattrs['linpars']
	labels = rawlabels[:]
	# Data
	data = []
	# Best-fit values
	truths = [] # blue vertical line in hist plot (dashed-black is median)
	iminssr = pardict['ssr'].argmin() # index of min ssr
	# Range for parameter (x) axis
	extents = []
	for i,p in enumerate(parlist):
		if p in linpars:
			data.append( pardict[p] )
		else:
			data.append( numpy.log10( pardict[p] ) )
			labels[i] = r'log$_{10}$ '+labels[i]
		truths.append( data[-1][iminssr] )
	data = numpy.vstack( data ).T
	from .emcee import triangle as dfmtriangle
	fig = dfmtriangle.corner(data, labels=labels, truths=truths) 
	# Now add a histogram for SSR
	x = pardict['ssr']
	ax = fig.axes[len(parlist)-2]
	ax.set_visible(True)
	ax.set_frame_on(True)
	ax.set_title('SSR')
	ax.set_yticklabels([])
	tbins = numpy.linspace(x.min(),x.min()+3.5*(numpy.median(x)-x.min()),50)
	ax.hist(x, bins=tbins, normed=True, color='black', histtype='step')
	return fig


def choose_bins_n_weights(x, bins, linear=False):
	whm = numpy.percentile(x,[15.87,50,84.13])
	if linear:
		#tbins = numpy.linspace(5.0*whm[0]-4.0*whm[1],5.0*whm[2]-4.0*whm[1],bins)
		tbins = numpy.linspace(3.0*whm[0]-2.0*whm[1],4.0*whm[1]-3.0*whm[0],bins)
		weights = None
	else:
		whm = numpy.log10(whm)
		#tbins = numpy.logspace(5.0*whm[0]-4.0*whm[1],5.0*whm[2]-4.0*whm[1],bins)
		tbins = numpy.logspace(5.0*whm[0]-4.0*whm[1],6.0*whm[1]-5.0*whm[0],bins)
		weights = numpy.ones_like(x)/len(x)/math.log10(tbins[1]/tbins[0])
	return tbins, weights


def hist_grid( keys, chainfiles, colors, dims=None, labels=None, bins=50, relative=[], nburn=0, linpars=None ):
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
	pardicts = {key: [] for key in keys}
	bestfits = {key: [] for key in keys}
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
				if cfn == relative[cfn]:
					x = pardicts[key][cfn]/numpy.median( pardicts[key][cfn] )
				else:
					x = pardicts[key][cfn][-clen:]/pardicts[key][relative[cfn]][-clen:]
			else:
				x = pardicts[key][cfn]
			normed = True
			if isinstance(bins,int):
				tbins, weights = choose_bins_n_weights(x, bins, linear=(key in clinpars))
				if key in clinpars:
					normed = True
				else:
					normed = False
			else:
				tbins = bins
			# FIXME: if someone provided bins, then weights is unset
			n,b,p = ax.hist(x, bins=tbins, normed=normed, weights=weights, color=colors[cfn], histtype='stepfilled', linewidth=0.0, alpha=0.2)
			ax.add_patch( matplotlib.patches.Polygon(p[0].get_xy(), fill=False, alpha=None, edgecolor=colors[cfn], linewidth=2.0) )
			nmax = max(numpy.max(n),nmax)

		ax.set_title(labels[i])
		ax.yaxis.set_visible(False)
		ax.set_ylim(0, 1.1*nmax)
		if key not in clinpars:
			ax.set_xscale('log')
	return gridfig.fig
