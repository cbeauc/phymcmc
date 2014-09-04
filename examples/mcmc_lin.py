import math
import numpy
import phymbie.fits
NegInf = float('-inf')

class mylinmodel(object):
	def __init__(self, pvec, par):
		par.vector = pvec
		self.par = par.pardict
	def get_solution(self,data):
		return self.par['slope']*data[:,0]+self.par['yint']
	def get_residuals(self,data):
		return data[:,1] - self.get_solution(data)
	def get_ssr(self,data):
		return (self.get_residuals(data)**2.0).sum()


def check_params_validity(pvec,par):
	if min(pvec) < 0:
		return False
	return True


def lnprobSSR(pvec,par,data):
	if check_params_validity(pvec,par):
		ssr = mylinmodel(pvec,par).get_ssr(data)
		if not math.isnan( ssr ):
			return -ssr
	return NegInf


def add_biological_params( pardict ):
	# Additional biological quantities (could also be part of pardict instead)
	biodict = {'N': 1.2e6, 'Volume': 0.5}
	# Example of a computed biological parameter (add as many as you want)
	biodict['slotyint'] = pardict['slope']*pardict['yint']/biodict['Volume']
	return biodict


datlin = numpy.loadtxt("lin.dat")

# Would either load params or initialize and fit
pdic = dict(slope=1.39,yint=573.13)
pfit = ['slope','yint']
params = phymbie.fits.ParamStruct(pdic,pfit)


###### THIS IS WHERE I FIT ######
if True:
	# This is fitting of params
	bfvec,bfssr = phymbie.fits.perform_fit(mylinmodel,params,datlin,verbose=False,rep_fit=1)
	params.vector = bfvec


###### THIS IS MY MCMC AREA #####
chain_file = 'outputs/chain_lin.hdf5'
if True:
	import phymbie.mcmc
	# MCMC parameters
	mcpars = dict(
		nwalkers = 300,
		nsteps = 300,
		nburn = 100,
		lnpostfn = lnprobSSR,
		stepsize = 2.0, # emcee parameter "a" which defines walkers' "step size"
		par = params,
		# Parameters over which walkers will be linearly distributed initially
		# Note that all parameters will be walked in LINEAR space
		linpars = ['slope'],
		# Arguments (args) passed to lnpostfn such that lnpostfn(pvec,par,args)
		args = datlin,
		threads = 4
	)
	# Setting up the MCMC sampler
	mcsampler = phymbie.mcmc.MCSampler( **mcpars )
	# Performing the MCMC run
	mcsampler.run_mcmc( chain_file )


###### ADDING BACK BIOLOGICAL PARAMETERS ######
if False: # FIXME
	import phymbie.mcmc
	mcpardict, _, _ = phymbie.mcmc.load_mcmc_chain( chain_file )
	biodict = add_biological_params( mcpardict )
	phymbie.mcmc.add_biodict_to_hdf5_chain( biodict, chain_file )
