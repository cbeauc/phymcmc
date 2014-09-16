import math
import numpy
import phymbie.mcmc

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

class grid_plot(object):
	def __init__(self, (gridheight,gridwidth)):
		import matplotlib.pyplot
		self.gh = gridheight
		self.gw = gridwidth
		# Setup the figure looking nice
		self.fig = matplotlib.pyplot.figure()
		self.fig.set_size_inches(3*self.gw,2.8*self.gh)
		matplotlib.pyplot.subplots_adjust(hspace=0.35, wspace=0.2)

	def subaxes(self, idx):
		return matplotlib.pyplot.subplot2grid((self.gh,self.gw), (idx/self.gw,idx%self.gw))


def convergence( chain_file, nburn=-1, parlist=None ):
	pardict, chainattrs = phymbie.mcmc.load_mcmc_chain( chain_file, nburn=nburn )
	niter = chainattrs['filledlength']
	if parslist is None:
		parlist = chainattrs['parfit']
	npars = len(parlist)
	print([npars, niter])

	for key in parlist:
		print('Error: broken function. FIXME')
	# Construct parameter array to hasten calculations below
	pararray = numpy.zeros((N,M))
	for idx,key in enumerate(parlist):
		pararray[:,i] = pardict[key]

	#### Intra-chain mean and variance
	thetam = pararray.mean(axis=0)
	s2m = pararray.var(axis=0,ddof=1.0) # ddof=1.0 means x.sum/(N-1)
	#### Inter-chain mean and variance
	thetaAVG = thetam.mean()
	Bredo = niter*thetam.var(ddof=1.0)
	print('B parameter: %.4g' % Bredo)
	Wredo = s2m.mean()
	print('W parameter: %.4g' % Wredo)

	Vhat =  (niter-1.0)/niter*Wredo + Bredo/niter + Bredo/(npars*niter) 
	PSRF =  ( Vhat/Wredo )**0.5 
	Pmod =  (  ( (niter-1.0)/niter*Wredo + Bredo/niter )/ Wredo )**0.5
	print('Answer PSRF = %.4g ' % PSRF)
	print('Answer PSRFm = %.4g ' % PSRF) # FIXME: print same thing twice?

	################################
	#Now advanced calc:

	VarV1 = ((niter-1.0)/niter)**2 /npars* sp.tvar(s2m) 
	VarV2 = ((npars+1.0)/(niter*npars))**2 * 2.0/(npars-1)*Bredo**2
	VarV3 = 2.0*(npars+1.0)*(niter-1.0)/(niter**2*npars)*(niter+0.0)/npars * (np.cov(s2m, thetam**2, ddof=0)[0,1] - 2.0*thetaAVG*np.cov(s2m, thetam, ddof = 0)[0,1] )*npars/(npars-1)
	VarVtot = VarV1 + VarV2 + VarV3

	d = 2* Vhat**2/VarVtot
	print('Vhat : %.4g' % Vhat)
	print('Degrees of freedom : %.4g' % d)
	Rc = ((d+3)/(d+1) * Vhat/Wredo )
	CSRF = Rc**0.5
	print('Estimated Point CSRF (Rc^0.5)= %.4g ' % Rc**0.5)

	######Now do upper CI:
	Bdf = npars - 1
	Wdf = 2 *Wredo**2 / (sp.tvar(s2m)/npars)
	R2fixed= ( niter-1.0)/niter
	R2rand = (1.0 + 1.0/npars) * (1.0/niter) * (Bredo/Wredo)
	R2est = R2fixed + R2rand
	R2upper = R2fixed + sp.distributions.f.ppf(0.975, Bdf, Wdf) *R2rand
	dfADJ  = (d+3.0)/(d+1)
	UPCI = (dfADJ * R2upper)**0.5
	print('Upper CI CSRF (Rc^0.5)= %.4g ' % UPCI)
	return [Vhat, Wredo, CSRF, UPCI]


def triangle( parlist, rawlabels, chain_file, nburn=-1 ):
	pardict, chainattrs = phymbie.mcmc.load_mcmc_chain( chain_file, nburn=nburn )
	labels = rawlabels[:]
	# Data
	data = []
	# Best-fit values
	truths = [] # blue vertical line in hist plot (dashed-black is median)
	iminssr = pardict['ssr'].argmin() # index of min ssr
	# Range for parameter (x) axis
	extents = []
	for i,p in enumerate(parlist):
		if p in chainattrs['linpars']:
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
	ax = fig.axes[11]
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


def hist_grid( keys, chainfiles, colors, dims=None, labels=None, bins=50, relative=[], nburn=-1 ):
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
		pdic,chainattrs = phymbie.mcmc.load_mcmc_chain( cf, nburn=nburn )
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
				tbins, weights = choose_bins_n_weights(x, bins, linear=(key in chainattrs['linpars']))
				if key in chainattrs['linpars']:
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
		if key not in chainattrs['linpars']:
			ax.set_xscale('log')
	return gridfig.fig

