import os
import phymbie.plot

#datlin = numpy.loadtxt("lin.dat")
chain_file = os.path.abspath('outputs/chain_lin.hdf5')
#chain_file = os.path.abspath('outputs/bob.hdf5')

plotpars = ['slope','yint']
plotparlabels = {'slope': r'slope, $m$', 'yint': r'$y_\mathrm{int}$'}

# Making the triangle plots
if True:
	fig = phymbie.plot.triangle(plotpars, plotparlabels.values(), chain_file)
	fig.savefig('outputs/lin_triangle.pdf')

# Making the hist plots
if False:
	#fig = phymbie.plot.hist('yint', [chain_file], ['black'], title=plotparlabels['yint'])
	#fig.savefig('outputs/hist_yint.pdf')
	fig = phymbie.plot.hist('slope', [chain_file], ['black'], title=plotparlabels['slope'])
	fig.savefig('outputs/hist_slope.pdf')
	#fig = phymbie.plot.triangle(['slope'], ['slope'], chain_file)
	#fig.savefig('outputs/hist_tri_slope.pdf')

