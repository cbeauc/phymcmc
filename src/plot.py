import numpy
import phymbie.mcmc
### plotting STUFF
import matplotlib
matplotlib.use('Agg')

params = {  'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'axes.titlesize': 'medium',
            'axes.labelsize': 'medium',
            'legend.fontsize': 'medium',
            'font.family': 'serif',
            'font.serif': 'Computer Modern Roman',
            'font.size': 14,
            'text.usetex': True}
matplotlib.rcParams.update(params)


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
	print('Your chain contained %d accepted parameters.' % len(pardict['ssr']))
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
	import triangle as dfmtriangle
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
	import math
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


def hist( key, chainfiles, colors, title=None, fig=None, ax=None, bins=50, relative=[], nburn=-1 ):
	import matplotlib.pyplot
	if fig is None:
		fig = matplotlib.pyplot.figure()
		if ax is None:
			fig.set_size_inches(3,3)
	if ax is None:
		ax = matplotlib.pyplot.axes()
	bestfit = []
	nmax = 0.0
	for i,cf in enumerate(chainfiles):
		pardict, chainattrs = phymbie.mcmc.load_mcmc_chain( cf, nburn=nburn )
		bestfit.append( numpy.median(pardict[key]) )
		if len(relative):
			try:
				x = pardict[key]/bestfit[relative[i]]
			except ValueError:
				raise ValueError("If you want to plot B relative to A then B should come after A in the list.")
		else:
			x = pardict[key]
		normed = True
		if isinstance(bins,int):
			tbins, weights = choose_bins_n_weights(x, bins, linear=(key in chainattrs['linpars']))
			if key in chainattrs['linpars']:
				normed = True
			else:
				normed = False
		else:
			tbins = bins
		n,b,p = ax.hist(x, bins=tbins, normed=normed, weights=weights, color=colors[i], histtype='stepfilled', linewidth=0.0, alpha=0.2)
		ax.add_patch( matplotlib.patches.Polygon(p[0].get_xy(), fill=False, alpha=None, edgecolor=colors[i], linewidth=2.0) )
		nmax = max(numpy.max(n),nmax)
		#ax.axvline(bestfit[i], color=colors[i], linewidth=1.5)
		#### FAKE STUFF
		#n,b,p = matplotlib.pyplot.hist(x*1.5, bins=bins, normed=True, color='red', histtype='stepfilled', linewidth=0, alpha=0.1)
		#ax.add_patch( matplotlib.patches.Polygon(p[0].get_xy(), fill=False, alpha=None, edgecolor='red', linewidth=2.0) )
		#nmax = max(numpy.max(n),nmax)
		#ax.axvline(numpy.median(x*1.5), color='red', linewidth=1.5)
		#### FAKE STUFF
	if title:
		ax.set_title(title)
	ax.yaxis.set_visible(False)
	ax.set_ylim(0, 1.1*nmax)
	if key not in chainattrs['linpars']:
		ax.set_xscale('log')
	return fig


def hist_grid( nup, keys, chainfiles, colors, titles=None, bins=50, relative=[], nburn=-1 ):
	print('fixme')
