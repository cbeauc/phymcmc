# Copyright (C) 2014-2020 Catherine Beauchemin <cbeau@users.sourceforge.net>
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

#
# =============================================================================
#
#                                   Utilities
#
# =============================================================================
#


def lnprobfn(pvec,model):
	return model.get_lnprob(pvec)


def restart_sampler( chain_file, model, pool=None, verbose=True ):
	# FIXME: should work but never been tested
	#	args should contain fitting data, for example!
	# NOTE: will only work if filledlength less than nwalkers*(nstep+1)
	# In the future, I could re-write to allow resizing of chain_file array
	f = h5py.File( chain_file, "r" )
	mcchain = f['mcchain']
	mcchaincopy = mcchain[:]
	# Set sampler parameters from chain_file
	mcpars = dict(
		chain_file = chain_file,
		model = model,
		nwalkers = mcchain.attrs['nwalkers'],
		nsteps = mcchain.attrs['nsteps'],
		linpars = eval(mcchain.attrs['linpars']),
		moves = mcchain.attrs['moves'],
		pool = pool,
		verbose = verbose
	)
	# Grab initialized sampler
	sampler = MCSampler( **mcpars )
	sampler.acceptance_fraction = f['acceptance_fraction'][:]
	# Now re-position your walkers at their last location
	idx = mcchain.attrs['filledlength']-mcchain.attrs['nwalkers']
	sampler.curlnprob = mcchaincopy[idx:,0]
	sampler.curpos = mcchaincopy[idx:,1:]
	f.close()
	return sampler


class MCSampler( object ):
	def __init__(self, chain_file, model, nwalkers, nsteps, linpars=[], moves=None, pool=None, verbose=True):
		# Required arguments
		self.chain_file = chain_file
		self.model = model
		self.nwalkers = nwalkers
		self.nsteps = nsteps
		# Optional arguments
		self.linpars = linpars
		self.moves = moves
		self.pool = pool
		self.verbose = verbose

		# 'moves' is specificed as string
		# this it to avoid clients needing to import emcee and
		# to allow storage of stepping method into hdf5 file.
		if self.moves is None:
			self.moves = 'emcee.moves.StretchMove(a=2.0)'

		# Additional parameters/properties of sampler
		self.npars = len(self.model.params.parfit)
		self.acceptance_fraction = []

		# Acquire the emcee sampler
		self.sampler = emcee.EnsembleSampler(self.nwalkers, self.npars, lnprobfn, moves=eval(self.moves), args=[self.model], pool=self.pool)


	def create_curlnprob(self, curpos):
		self.curlnprob = numpy.zeros( self.nwalkers )
		for idx,pcandidate in enumerate(curpos):
			lprob = lnprobfn(pcandidate,self.model)
			# Sanity-check the candidate positions
			assert not math.isinf(lprob), 'A provided parameter set gave infinite lnprob on row %d:\n par=%s.' % (idx,str(pcandidate))
			assert not math.isnan(lprob), 'A provided parameter set gave NaN lnprob on row %d:\n par=%s.' % (idx,str(pcandidate))
			self.curlnprob[idx] = lprob
			if self.verbose:
				print('# Accepted walker: %d (lnprob=%g)' % (idx,lprob))
				print( ('%g '*self.npars) % tuple(pcandidate) )
				sys.stdout.flush()


	def init_walkers_for_me(self, linbw=0.5, logbw=1.0, minlnprob=1.e20):
		self.tstart = time.time()
		# Check for bad values entered
		assert 0.0 < linbw < 1.0, "ERROR: Walker initialization parameter linbw providing the width of the linear box must be in (0,1)"
		# Position all your walkers
		if self.verbose:
			print('Positioning the walkers randomly-uniformly for you')
			sys.stdout.flush()
		# Initial position array has dimensions (nwalker, nparams)
		self.curpos = numpy.zeros( (self.nwalkers, self.npars) )
		self.curlnprob = numpy.zeros( self.nwalkers )
		# Walker 0 gets started at the best fit position (centre)
		self.curpos[0,:] = self.model.params.vector
		self.curlnprob[0] = lnprobfn(self.model.params.vector,self.model)
		if self.verbose:
			print('# Accepted walker: 0 (lnprob=%g)' % self.curlnprob[0])
			print( ('%g '*self.npars) % tuple(self.model.params.vector) )
			sys.stdout.flush()
		# Remaining walkers are distributed randomly, uniformly (lin or log)
		wrem = self.nwalkers-1
		bfcentrevec = numpy.array(self.model.params.vector)
		while wrem:
			# Generate a candidate position for a walker
			pcandidate = bfcentrevec.copy()
			for pi,pname in enumerate(self.model.params.parfit):
				if pname in self.linpars:
					pcandidate[pi] *= random.uniform(1.0-linbw,1.0+linbw)
				else:
					pcandidate[pi] *= 10.0**random.uniform(-logbw,logbw)
			# Accept or reject the candidate position
			lprob = lnprobfn(pcandidate,self.model)
			if not math.isinf(lprob) and not math.isnan(lprob):
				if (minlnprob == 1.e20) or (lprob > minlnprob):
					self.curlnprob[wrem] = lprob
					self.curpos[wrem,:] = pcandidate
					if self.verbose:
						print('# Accepted walker: %d (lnprob=%g)' % (self.nwalkers-wrem,lprob))
						print( ('%g '*self.npars) % tuple(pcandidate) )
						sys.stdout.flush()
					wrem -= 1


	def init_walkers_from_chain(self, oldchainfile, replace_stuck_walkers=False, conditions=None, newpars=None):
		self.tstart = time.time()
		if self.verbose:
			print('Reading walkers initial pos from end of %s'% oldchainfile)
			sys.stdout.flush()
		f = h5py.File(oldchainfile, 'r')
		mcchain = f['mcchain']
		mcchaincopy = mcchain[:]
		# Make sure the # walkers in old chain match what's requested
		assert mcchain.attrs['nwalkers'] == self.nwalkers, 'The number of walkers in %s (%d) is not what you requested (%d).' % (oldchainfile,mcchain.attrs['nwalkers'],self.nwalkers)
		# Keep a copy of the old list of params to estimate
		oldparfit = list(eval(mcchain.attrs['parfit']))[1:]
		# Grab walker's position on their last step
		iend = mcchain.attrs['filledlength']
		nwalkers = mcchain.attrs['nwalkers']
		f.close()
		if (conditions is None) and not replace_stuck_walkers:
			curposs = mcchaincopy[iend-nwalkers:iend,1:]
		else:
			if conditions is None: # apply no conditions
				conditions = lambda lam,pval,pkey: True
			mcchaincopy = mcchaincopy[iend-1::-1,:] # start from end of chain
			# Identify the bad walkers
			if replace_stuck_walkers:
				mmv = numpy.array([numpy.diff(numpy.nonzero(numpy.ediff1d(mcchaincopy[a::nwalkers,0],to_begin=1))[0]).mean() for a in range(nwalkers)])
				badwalks = numpy.nonzero(mmv>100.0)[0]
			else:
				badwalks = ()
			# Form array of suitable walkers
			icur = -1
			curposs = []
			while len(curposs) < nwalkers:
				icur += 1
				candidate = mcchaincopy[icur]
				if (icur % nwalkers) in badwalks:
					print('# Replaced stuck walker\n', repr(candidate))
					continue
				if conditions(candidate[0],candidate[1:],oldparfit):
					curposs.append( candidate[1:] )
				else:
					print('# Replaced bad walker\n', repr(candidate))
			curposs = numpy.array(curposs)
			sys.stdout.flush()
		# Make sure the params in oldchain match what's requested or add them
		self.curpos = []
		for par in self.model.params.parfit:
			if par in oldparfit:
				self.curpos.append( curposs[:,oldparfit.index(par)] )
			else:
				assert newpars is not None, 'Param \'%s\' is not in oldchain but is required in newchain. Provide via newpars argument.' % par
				self.curpos.append( newpars[par] )
		self.curpos = numpy.vstack( self.curpos ).T
		# Re-compute lnprob (best not to trust hdf5 chain file, in case)
		self.create_curlnprob( self.curpos )


	def init_walkers_from_array(self,curpos,lnprob=None):
		"""
			Initialize position of walkers in mcmc chain from arrays."
			Input:
				curpos (required) = current position of walkers
				lnprob (optional) = ln likelihood probability
			The lenght of lnprob and curpos must match.
			The number of columns and parameter order in curpos should match
				that in the provided MCSampler's par.parfit list.
			If lnprob not provided, it is computed for you.
		"""
		self.tstart = time.time()
		if self.verbose:
			print('Reading walkers initial position from array...')
			sys.stdout.flush()
		# Make sure the # walkers and # params in curpos array match request
		assert self.nwalkers == curpos.shape[0], 'The number of walkers (lines) in curpos (%d) does not match nwalkers requested (%d).' % (len(curpos),self.nwalkers)
		assert self.npars == curpos.shape[1], 'The number of parametres in curpos (%d) does not match npars in params (%d).' % (curpos.shape[1],self.npars)
		self.curpos = curpos.copy() # accept positions provided
		# Check if lnprob given, create if not provided
		if lnprob is None:
			self.create_curlnprob( self.curpos )
		else: # if given, check if correct size
			assert self.nwalkers == len(lnprob), 'The number of walkers (lines) in lnprob (%d) does not match nwalkers requested (%d).' % (len(lnprob),self.nwalkers)
			self.curlnprob = numpy.array( lnprob )
			# Print walkers initial position for user (if verbose)
			if self.verbose:
				for idx,(lprob,pos) in enumerate(zip(lnprob,curpos)):
					print('# Accepted walker: %d (lnprob=%g)' % (idx,lprob))
					print( ('%g '*self.npars) % tuple(pos) )
					sys.stdout.flush()


	def init_chainfile(self):
		# Initialize the chain file
		# Open/create the HDF5 chain file
		f = h5py.File(self.chain_file, "w")
		# Create the "flatchain" structure to hold the chain array
		fchain = f.create_dataset("mcchain", ((self.nsteps+1)*self.nwalkers,self.npars+1), dtype='float64')
		# Store some additional properties along with the chain
		fchain.attrs['filledlength'] = 0
		fchain.attrs['pardict'] = '{\'lnprob\': 0.0, '+repr(self.model.params.pardict)[1:]
		fchain.attrs['parfit'] = repr(['lnprob']+self.model.params.parfit)
		fchain.attrs['nwalkers'] = self.nwalkers
		fchain.attrs['nsteps'] = self.nsteps
		fchain.attrs['linpars'] = repr(self.linpars)
		fchain.attrs['moves'] = self.moves
		# Store the walker's original position into the chain file
		nl = self.nwalkers
		f['mcchain'][:nl,0] = self.curlnprob
		f['mcchain'][:nl,1:] = self.curpos
		f['mcchain'].attrs['filledlength'] = nl
		f.close()
		if self.verbose:
			print('Initialization took %g min\n' % ((time.time()-self.tstart)/60.0))
			sys.stdout.flush()


	def run_mcmc(self, chunk_size=30000):
		self.init_chainfile()
		if self.verbose:
			print('Starting the MCMC run...')
			sys.stdout.flush()
			trunstart = time.time()
		# initialize the sampler
		initial_state = emcee.State(self.curpos, log_prob=self.curlnprob)
		# Now walk...
		poss = []
		lnprobs = []
		twrite = time.time()
		for nstp, ((nwpos, nwlnprob, _),nacc) in enumerate(self.sampler.sample(initial_state, iterations=self.nsteps, store=False)):

			# Store current pos + lnprob of each walker
			poss.append(nwpos.copy())
			lnprobs.append(nwlnprob.copy())

			# Average fraction of accepted since start
			#	averaged over all walkers (ideally in 0.2-0.5 range).
			self.acceptance_fraction.append( numpy.mean(nacc) )

			# Print to file every once in a while
			if (len(lnprobs)*self.nwalkers > chunk_size) or nstp==self.nsteps-1:
				if self.verbose:
					print('   accepting %d params took %g min' % (len(lnprobs)*self.nwalkers,(time.time()-twrite)/60.0))
					sys.stdout.flush()
					self.tstart = time.time()
				lnprobs = numpy.array(lnprobs).flatten()
				nl = len(lnprobs)
				f = h5py.File(self.chain_file, "r+")
				s = f['mcchain'].attrs['filledlength']
				f['mcchain'][s:s+nl,0] = lnprobs
				f['mcchain'][s:s+nl,1:] = numpy.array(poss).reshape((nl,self.npars))
				f['mcchain'].attrs['filledlength'] = s+nl
				# If acceptance_fraction already exists
				if 'acceptance_fraction' in f:
					del f['acceptance_fraction']
				f.create_dataset('acceptance_fraction', data=numpy.array(self.acceptance_fraction))
				# All done updating hdf5 file
				f.close()
				if self.verbose:
					twrite = time.time()
					print('   writing to disk took %g s' % (twrite-self.tstart))
					print('Wrote %g%% of simulation to disk for you.' % (100*(nstp+1)/self.nsteps))
					sys.stdout.flush()
				poss = []
				lnprobs = []
		if self.verbose:
			print('The complete MCMC run took %g h. Enjoy!' % ((time.time()-trunstart)/3600.0))


def load_mcmc_chain( chain_file, nburn=0, asdict=True, verbose=True ):
	f = h5py.File( chain_file, "r" )
	mcchain = f['mcchain']
	# Copy over only the non-zero, filled part of array
	mcchaincopy = mcchain[:mcchain.attrs['filledlength'],:]
	pardict = eval( mcchain.attrs['pardict'] )
	# Build dictionary of chain attributes
	chainattrs = {}
	chainattrs['parfit'] = eval(mcchain.attrs['parfit'])
	chainattrs['acceptance_fraction'] = f['acceptance_fraction'][:]
	chainattrs['filledlength'] = mcchain.attrs['filledlength']
	chainattrs['nwalkers'] = mcchain.attrs['nwalkers']
	chainattrs['nsteps'] = mcchain.attrs['nsteps']
	chainattrs['linpars'] = eval(mcchain.attrs['linpars'])
	chainattrs['moves'] = mcchain.attrs['moves']
	# If parameters derived/calculated from the chain are in there
	# get them out as well.
	derivedchain = False
	if 'derivedchain' in f:
		derivedchain = f['derivedchain'][:]
		chainattrs['parderived'] = f['derivedchain'].attrs['parlist']
		pardict.update( eval(f['derivedchain'].attrs['pardict']) )
	f.close()
	# nburn can be a fraction of the run rather than a number of steps
	# default is zero burn-steps
	if 0.0 < nburn < 1.0:
		chainattrs['nburn'] = int(round(chainattrs['filledlength']/chainattrs['nwalkers']*nburn))
	else:
		chainattrs['nburn'] = nburn
	idx = numpy.arange(chainattrs['nburn']*chainattrs['nwalkers'],chainattrs['filledlength'])
	# Slice through the two array attributes based on nburn
	chainattrs['acceptance_fraction'] = chainattrs['acceptance_fraction'][nburn:]
	max_lnprob_row = mcchaincopy[:chainattrs['filledlength'],0].argmax()
	if verbose and (max_lnprob_row not in idx):
		print('WARNING: Your max lnprob was discarded when the burn-in was applied.')
	# And now make my dictionary of parameters
	if asdict:
		for pi,pn in enumerate(chainattrs['parfit']):
			pardict[pn] = mcchaincopy[idx,pi]
		# Add the derived parameters to the dictionary if they exist
		if derivedchain is not False:
			for pi,pn in enumerate(chainattrs['parderived']):
				pardict[pn] = derivedchain[idx,pi]
		# Tell people about what you got for them ;)
		if verbose:
			print('Your chain contained %d accepted parameters.' % len(pardict['lnprob']))
		return pardict, chainattrs
	else:
		chainattrs['pardict'] = pardict
		if derivedchain is not False:
			return numpy.hstack((mcchaincopy[idx,:],derivedchain[idx,:])), chainattrs
	return mcchaincopy[idx,:], chainattrs 


def load_mcmc_bestfit( chain_file, verbose=False, nburn=0 ):
	opdic,pattrs = load_mcmc_chain( chain_file, nburn=nburn )
	pfit = pattrs['parfit']
	idx = opdic['lnprob'].argmax()
	pdic = {}
	for key,val in opdic.items():
		if isinstance(val, numpy.ndarray):
			pdic[key] = val[idx]
		else:
			pdic[key] = val
	pfit = list(pfit[1:])
	if verbose:
		print('Your best param set (lnprob = %g) was:' % pdic['lnprob'])
		print(repr(pfit))
		print(repr(pdic))
	return (pdic,pfit)


def add_derived_dict_to_mcmc_chain( derivedparfn, chain_file ):
	# load the current chain
	pdic, chainattrs = load_mcmc_chain( chain_file, nburn=0 )
	# compute and obtain the derived dictionary
	deriveddic = derivedparfn( pdic )
	# put the resulting dictionary in mcmc-hdf5 friendly format
	derivedpardict = dict() # 1st row values of all derived params
	derivedparlist = [] # derived params that are arrays
	derivedarray = [] # array of derived params that are arrays
	for key,value in deriveddic.items():
		if type(value) is float:
			derivedpardict[key] = value
		else:
			derivedpardict[key] = value[0]
			derivedparlist.append( key )
			derivedarray.append( value )
	derivedarray = numpy.array(derivedarray).T
	f = h5py.File(chain_file, 'r+')
	# If a derivedchain already exists, delete it before proceeding
	if 'derivedchain' in f:
		del f['derivedchain']
	# Create the chain structure to hold the derived chain array
	fderivedchain = f.create_dataset('derivedchain', data=derivedarray)
	f['derivedchain'].attrs['parlist'] = derivedparlist
	f['derivedchain'].attrs['pardict'] = repr(derivedpardict)
	f.close()

