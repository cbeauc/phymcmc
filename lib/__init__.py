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

# =============================================================================

from __future__ import print_function
NegInf = float('-inf')

class ParamStruct(object):
	def __init__(self,pardict,parfit):
		self.pardict = pardict
		self.parfit = parfit
	def validate(self):
		pass
	@property
	def vector(self):
		return tuple(self.pardict[key] for key in self.parfit)
	@vector.setter
	def vector(self,fittedpars):
		for key,val in zip(self.parfit,fittedpars):
			self.pardict[key] = val
		self.validate()


class base_model(object):
	def __init__(self, data, params):
		self.data = data
		self.params = params
	def get_normalized_ssr(self,pvec):
		""" Computes total normalized SSR, i.e. SSR/stdev. """
		raise NotImplementedError
	def get_lnprob(self,pvec):
		"""
			Determine the lnprob for the model, given the parameters.
			for running a log parameter in lin scale or whatever). But
			DO NOT check here for NaN. The sanity parsing of the SSR value
			is done by the lnprobfn function in the phymcmc MCMC library.
		"""
		try:
			nssr = self.get_normalized_ssr(pvec)
		except ValueError: # return if params are invalid
			return NegInf
		# IF we get a NaN...
		# We should NOT ignore this since passing -inf is equivalent to
		#   forbidding this parameter set value. So this is an additional
		#   constraint we place on our parameters that we MUST be told
		#   about. When you encounter this warning, investigate!
		import math
		if math.isnan( nssr ):
			print('WARNING: lnprob encountered NaN SSR (and returned -inf) for:\n '+repr(self.params.pardict), file=sys.stderr)
			return NegInf
		return -0.5*nssr


def odeint(*args,**kwargs):
	import scipy.integrate
	assert 'mxstep' not in kwargs
	kwargs['mxstep'] = 4000000
	return scipy.integrate.odeint(*args,**kwargs)
