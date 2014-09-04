import phymbie.plot
import phymbie.model

#datlin = numpy.loadtxt("lin.dat")
strains = ['H275', 'Y275', 'I223', 'V223']
sdir = '/tmp/Fits/'
chain_files = [sdir+strn+'/'+strn+'_chain.hdf5' for strn in strains]

plotpars = ['cpfu','tE', 'tI', 'b', 'vom', 'pfm', 'pfs', 'pr', 'p2r']
#plotpars = ['cpfu','tE', 'tI', 'b', 'pr', 'p2r']
#plotparlabels = [phymbie.model.parsymbs[key] for key in plotpars]

# Making the triangle plots
if True:
	for si,strain in enumerate(strains):
		plotparlabels = plotpars
		fig = phymbie.plot.triangle(plotpars, plotparlabels, chain_files[si])
		fig.savefig('realouts/'+strain+'_triangle.png')

# Making individual, relative hist plots
if False:
	colorlist = ['black','red','blue','green']
	relative = [0, 0, 2, 2]
	for p in ['tE']:
		parlabel = phymbie.model.parnames[p]+', '+phymbie.model.parsymbs[p]
		fig = phymbie.plot.hist(p, chain_files, colorlist, title=parlabel)
		fig.savefig('realouts/hist_'+p+'.pdf')

# Making a 3x2 grid of absolute hist plots
if False:
	strains = ['H275', 'Y275', 'I223', 'V223']
	chain_files = [sdir+strn+'/'+strn+'_chain.hdf5' for strn in strains]
	relative = [0, 0, 2, 2]
	colorlist = ['black','red','blue', 'green']
	plotpars = ['cpfu', 'tE', 'tI', 'tinf', 'R0', 'burst', 'prcell', 'pfs2fm']
	parlabels = [r'Inf.\ clearance, $c_\mathrm{pfu}$ (1/h)', r'Eclipse phase, $\tau_E$ (h)', r'Infectious lifespan, $\tau_I$ (h)', r'Infecting time, $t_\mathrm{inf}$ (min)', r'Basic repro.\ num., $R_0$', r'RNA burst size (RNA/cell)', r'Prod.\ rate (RNA/h/cell)', r'PFU prod.\ ratio, $p_\mathrm{SC}/p_\mathrm{MC}$']
	#r'Eclipse phase, $\tau_E$ (h)', r'Prod.\ rate (RNA/h/cell)', r'Infectious lifespan, $\tau_I$ (h)', r'$R_0$/burst (infection/RNA)', r'Infecting time, $t_\mathrm{inf}$ (min)', r'Basic repro.\ num., $R_0$']
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot
	import numpy
	gw = 4
	gh = 2
	fig = matplotlib.pyplot.figure()
	fig.set_size_inches(3*gw,2.8*gh)
	matplotlib.pyplot.subplots_adjust(hspace=0.35, wspace=0.2)
	for i,p in enumerate(plotpars):
		ax = matplotlib.pyplot.subplot2grid((gh,gw), (i/gw,i%gw))
		phymbie.plot.hist(p, chain_files, colorlist, fig=fig, ax=ax, title=parlabels[i], relative=relative)
	fig.savefig('realouts/hist_eric_grid.pdf', bbox_inches='tight')

