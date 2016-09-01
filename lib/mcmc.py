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

from __future__ import print_function 
from phymcmc import emcee
import h5py
import math
import numpy
import random
import sys
import time
NegInf = float('-inf')

#
# =============================================================================
#
#                                   Utilities
#
# =============================================================================
#


def lnprobfn(pvec,model,*args,**kwargs):
	lnprob = model.get_lnprob(pvec,*args,**kwargs)
	# IF we get a NaN...
	# We should NOT ignore this since passing -inf is equivalent to
	#   forbidding this parameter set value. So this is an additional
	#   constraint we place on our parameters that we MUST be told
	#   about. When you encounter this warning, investigate!
	if math.isnan( lnprob ):
		print('WARNING: lnprob encountered NaN SSR (and returned -inf) for:\n '+repr(pvec), file=sys.stderr)
		return NegInf
	return lnprob


def restart_sampler( chain_file, model, args=None, threads=1, pool=None, verbose=True ):
	# FIXME: should work but never been tested
	#	args should contain fitting data, for example!
	# NOTE: will only work if filledlength less than nwalkers*(nstep+1)
	# In the future, I could re-write to allow resizing of chain_file array
	f = h5py.File( chain_file, "r" )
	mcchain = f['mcchain']
	mcchaincopy = mcchain.value
	# Recuperate best-fit params
	pdic = eval( mcchain.attrs['pardict'] )
	pfit = mcchain.attrs['parfit'][1:] # discard SSR as a fitting parameter
	params = phymcmc.fits.ParamStruct(pdic,pfit)
	# Set sampler parameters from chain_file
	mcpars = dict(
		chain_file = chain_file,
		model = model,
		nwalkers = mcchain.attrs['nwalkers'],
		nsteps = mcchain.attrs['nsteps'],
		par = params,
		stepsize = mcchain.attrs['stepsize'],
		linbw = mcchain.attrs['linbw'],
		logbw = mcchain.attrs['linbw'],
		linpars = mcchain.attrs['linpars'],
		args = args,
		threads = threads,
		pool = pool,
		verbose = verbose,
		restart_from_file = True
	)
	# Grab initialized sampler
	sampler = MCSampler( **mcpars )
	sampler.acceptance_fraction = mcchain.attrs['acceptance_fraction']
	sampler.acor = f['autocorr'].value
	# Now re-position your walkers at their last location
	idx = mcchain.attrs['filledlength']-mcchain.attrs['nwalkers']
	sampler.curlnprob = -mcchaincopy[idx:,0]
	sampler.curpos = mcchaincopy[idx:,1:]
	f.close()
	return sampler


class MCSampler( object ):
	def __init__(self, chain_file, model, nwalkers, nsteps, par, stepsize=2.0, linbw=0.5, logbw=1.0, linpars=[], args=None, threads=1, pool=None, maxssr=1.0e20, verbose=True, restart_from_file=False):
		# Required arguments
		self.chain_file = chain_file
		self.model = model
		self.nwalkers = nwalkers
		self.nsteps = nsteps
		self.par = par
		# Optional arguments
		self.stepsize = stepsize
		self.linbw = linbw
		self.logbw = logbw
		self.linpars = linpars
		self.args = args if args is not None else ()
		self.threads = threads
		self.pool = pool
		self.maxssr = maxssr
		self.verbose = verbose

		# Additional parameters/properties of sampler
		self.npars = len(self.par.parfit)
		self.acceptance_fraction = []
		self.acor = []

		# Check for bad values entered
		assert 0.0 < self.linbw < 1.0, "MCMC parameter linboxwidth must be in (0,1)"

		# Acquire the emcee sampler
		self.sampler = emcee.EnsembleSampler(self.nwalkers, self.npars, lnprobfn, a=self.stepsize, args=(self.model,self.par,self.args), threads=self.threads, pool=self.pool)

		if not restart_from_file:
			tstart = time.time()
			self.initialize_the_walkers()
			self.initialize_the_chainfile()
			if self.verbose:
				print('Initialization took %g min\n' % ((time.time()-tstart)/60.0))


	def initialize_the_walkers(self,oldchainfile=None):
		if oldchainfile is None:
			# Position all your walkers
			if self.verbose:
				print('Positioning the walkers')
			# the initial position array has dimensions (nwalker, nparams)
			self.curpos = numpy.zeros( (self.nwalkers, self.npars) )
			self.curlnprob = numpy.zeros( self.nwalkers )

			# walker 0 gets started at the best fit position (centre)
			self.curpos[0,:] = self.par.vector
			self.curlnprob[0] = lnprobfn(self.par.vector,self.model,self.par,self.args)
			if self.verbose:
				print('# Accepted walker: 0 (ssr=%g)' % -self.curlnprob[0])
				print( ('%g '*self.npars) % tuple(self.par.vector) )

			# the remaining walkers are distributed randomly, uniformly (lin or log)
			wrem = self.nwalkers-1
			bfcentrevec = numpy.array(self.par.vector)
			while wrem:
				# generate a candidate position for a walker
				pcandidate = bfcentrevec.copy()
				for pi,pname in enumerate(self.par.parfit):
					if pname in self.linpars:
						pcandidate[pi] *= random.uniform(1.0-self.linbw,1.0+self.linbw)
					else:
						pcandidate[pi] *= 10.0**random.uniform(-self.logbw,self.logbw)
				# accept or reject the candidate position
				lprob = lnprobfn(pcandidate,self.model,self.par,self.args)
				if not math.isinf( lprob ):
					if (self.maxssr == 1.0e20) or (-lprob < self.maxssr):
						self.curlnprob[wrem] = lprob
						self.curpos[wrem,:] = pcandidate
						if self.verbose:
							print('# Accepted walker: %d (ssr=%g)' % (self.nwalkers-wrem,-lprob))
							print( ('%g '*self.npars) % tuple(pcandidate) )
							sys.stdout.flush()
						wrem -= 1
		else:
			if self.verbose:
				print('Reading walkers initial pos from end of %s'% oldchainfile)
			f = h5py.File(oldchainfile, 'r')
			mcchain = f['mcchain']
			mcchaincopy = mcchain.value
			# Make sure the # walkers in old chain match what's requested
			assert mcchain.attrs['nwalkers'] == self.nwalkers, 'The number of walkers in %s (%d) is not what you requested (%d).' % (oldchainfile,mcchain.attrs['nwalkers'],self.nwalkers)
			# Now re-position your walkers at their last location
			idf = mcchain.attrs['filledlength']
			idi = idf-mcchain.attrs['nwalkers']
			self.curlnprob = -mcchaincopy[idi:idf,0]
			self.curpos = mcchaincopy[idi:idf,1:]
			f.close()


	def initialize_the_chainfile(self):
		# Initialize the chain file
		# Open/create the HDF5 chain file
		f = h5py.File(self.chain_file, "w")
		# Create the "flatchain" structure to hold the chain array
		fchain = f.create_dataset("mcchain", ((self.nsteps+1)*self.nwalkers,self.npars+1))
		# Store some additional properties along with the chain
		fchain.attrs['filledlength'] = 0
		fchain.attrs['pardict'] = '{\'ssr\': 0.0, '+repr(self.par.pardict)[1:]
		fchain.attrs['parfit'] = ['ssr']+self.par.parfit
		fchain.attrs['nwalkers'] = self.nwalkers
		fchain.attrs['nsteps'] = self.nsteps
		fchain.attrs['stepsize'] = self.stepsize
		fchain.attrs['linbw'] = self.linbw
		fchain.attrs['logbw'] = self.logbw
		fchain.attrs['linpars'] = self.linpars
		# Store the walker's original position into the chain file
		nl = self.nwalkers
		f['mcchain'][:nl,0] = -self.curlnprob
		f['mcchain'][:nl,1:] = self.curpos
		f['mcchain'].attrs['filledlength'] = nl
		f.close()

	def run_mcmc(self, chunk_size=30000):
		if self.verbose:
			print('Starting the MCMC run...')
			trunstart = time.time()
		# Now walk...
		poss = []
		lnprobs = []
		meanpos = [ numpy.mean(self.curpos,axis=0) ]
		twrite = time.time()
		for nstp, (pos, lnprob, _) in enumerate(self.sampler.sample(self.curpos, lnprob0=self.curlnprob, iterations=self.nsteps, storechain=False)):

			# Store current pos + lnprob of each walker
			poss.append(pos*1.0)
			lnprobs.append(lnprob*1.0)

			# Average fraction of accepted since start
			#	averaged over all walkers (ideally in 0.2-0.5 range).
			self.acceptance_fraction.append( numpy.mean(self.sampler.acceptance_fraction) )
			# Integrated autocorrelation time since start
			#	computed from average pos of all walkers at each step.
			#	emcee developer suggests a burn-in time of ~ 10x autocor time.
			meanpos.append(pos.mean(axis=0))
			self.acor.append(emcee.autocorr.integrated_time(numpy.array(meanpos)))

			# Print to file every once in a while
			if (len(lnprobs)*self.nwalkers > chunk_size) or nstp==self.nsteps-1:
				if self.verbose:
					print('   accepting %d params took %g min' % (len(lnprobs)*self.nwalkers,(time.time()-twrite)/60.0))
					tstart = time.time()
				lnprobs = numpy.array(lnprobs).flatten()
				nl = len(lnprobs)
				f = h5py.File(self.chain_file, "r+")
				s = f['mcchain'].attrs['filledlength']
				f['mcchain'][s:s+nl,0] = -lnprobs
				f['mcchain'][s:s+nl,1:] = numpy.array(poss).reshape((nl,self.npars))
				f['mcchain'].attrs['filledlength'] = s+nl
				f['mcchain'].attrs['acceptance_fraction'] = self.acceptance_fraction
				# If autocorr already exists, delete it before proceeding
				if 'autocorr' in f:
					del f['autocorr']
				# Create the chain structure to hold the autocorr array
				f.create_dataset('autocorr', data=numpy.array(self.acor))
				f.close()
				if self.verbose:
					twrite = time.time()
					print('   writing to dist took %g s' % (twrite-tstart))
					print('Wrote %g%% of simulation to disk for you.' % (100*(nstp+1)/self.nsteps))
				poss = []
				lnprobs = []
		if self.verbose:
			print('The complete MCMC run took %g h. Enjoy!' % ((time.time()-trunstart)/3600.0))


def load_mcmc_chain( chain_file, nburn=0 ):
	f = h5py.File( chain_file, "r" )
	mcchain = f['mcchain']
	# Copy over only the non-zero, filled part of array
	mcchaincopy = mcchain.value[:mcchain.attrs['filledlength'],:]
	pardict = eval( mcchain.attrs['pardict'] )
	# Build dictionary of chain attributes
	chainattrs = {}
	chainattrs['parfit'] = mcchain.attrs['parfit']
	chainattrs['acceptance_fraction'] = mcchain.attrs['acceptance_fraction']
	chainattrs['acor'] = f['autocorr'].value
	chainattrs['filledlength'] = mcchain.attrs['filledlength']
	chainattrs['nwalkers'] = mcchain.attrs['nwalkers']
	chainattrs['nsteps'] = mcchain.attrs['nsteps']
	chainattrs['stepsize'] = mcchain.attrs['stepsize']
	chainattrs['linbw'] = mcchain.attrs['linbw']
	chainattrs['logbw'] = mcchain.attrs['logbw']
	chainattrs['linpars'] = mcchain.attrs['linpars']
	# If parameters derived/calculated from the chain are in there
	# get them out as well.
	derivedchain = False
	if 'derivedchain' in f:
		derivedchain = f['derivedchain'].value
		derivedparlist = f['derivedchain'].attrs['parlist']
	f.close()
	# nburn can be a fraction of the run rather than a number of steps
	# default is zero burn-steps
	if 0.0 < nburn < 1.0:
		chainattrs['nburn'] = int(round(chainattrs['filledlength']/chainattrs['nwalkers']*nburn))
	else:
		chainattrs['nburn'] = nburn
	idx = numpy.arange(chainattrs['nburn']*chainattrs['nwalkers'],chainattrs['filledlength'])
	# Slice trhough the two array attributes based on nburn
	chainattrs['acceptance_fraction'] = chainattrs['acceptance_fraction'][nburn:]
	chainattrs['acor'] = chainattrs['acor'][nburn:,:]
	min_ssr_row = mcchaincopy[:chainattrs['filledlength'],0].argmin()
	if min_ssr_row not in idx:
		print('WARNING: Your best SSR was discarded when the burn-in was applied.')
	# And now make my dictionary of parameters
	for pi,pn in enumerate(chainattrs['parfit']):
		pardict[pn] = mcchaincopy[idx,pi]
	# Add the derived parameters to the dictionary if they exist
	if derivedchain is not False:
		for pi,pn in enumerate(derivedparlist):
			pardict[pn] = derivedchain[idx,pi]
	# Tell people about what you got for them ;)
	print('Your chain contained %d accepted parameters.' % len(pardict['ssr']))
	return pardict, chainattrs


def load_mcmc_bestfit( chain_file, verbose=False, nburn=0 ):
	opdic,pattrs = load_mcmc_chain( chain_file, nburn=nburn )
	pfit = pattrs['parfit']
	idx = opdic['ssr'].argmin()
	pdic = {}
	for key,val in opdic.items():
		try:
			pdic[key] = val[idx]
		except TypeError:
			pdic[key] = val
	pfit = pfit[1:]
	if verbose:
		print('Your best-fit was (ssr = %g)' % pdic['ssr'])
		print(repr(pfit))
		print(repr(pdic))
	return (pdic,pfit)


def add_derived_dict_to_mcmc_chain( derivedparfn, chain_file ):
	pdic, chainattrs = load_mcmc_chain( chain_file, nburn=0 )
	deriveddic = derivedparfn( pdic )
	derivedlist = deriveddic.keys()
	derivedarray = numpy.array(deriveddic.values()).T
	f = h5py.File(chain_file, 'r+')
	# If a derivedchain already exists, delete it before proceeding
	if 'derivedchain' in f:
		del f['derivedchain']
	# Create the chain structure to hold the derived chain array
	fderivedchain = f.create_dataset('derivedchain', data=derivedarray)
	f['derivedchain'].attrs['parlist'] = derivedlist
	f.close()

