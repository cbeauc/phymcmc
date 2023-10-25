# Copyright (C) 2014-2022 Catherine Beauchemin <cbeau@users.sourceforge.net>
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
import math
import sys
NegInf = float('-inf')


class ODEintegrationError(ValueError):
	"""Raised if ODE integration is unsuccessful."""
	pass


class ParamStruct(object):
	def __init__(self,pardict,parfit):
		self.pardict = pardict
		self.parfit = parfit
		self.validate()
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
	def ln_probability(self,pvec):
		"""
			Computes the numerator of Bayes' theorem
			   ln[probability] = ln[likelihood] + ln[prior]
			where
				ln[ likelihood ] = ln[ Likeli(params|model,data) ]
					lnlikeli = -SSR/(2*sigma^2)
				ln[ prior ] = ln[ Prior(params) ]
					lnprior = 0.0+0.0+0.0    (linear uniform pars)
					lnprior = -ln[ p1*p2*p3 ]   (log uniform pars)
		"""
		raise NotImplementedError
	def get_lnprob(self,pvec):
		"""
			Runs lnprob provided by the model given the parameters,
			and handles typical errors raised.
		"""
		# Handle unexpected model runtime calculation errors
		try:
			lnprob = self.ln_probability(pvec)
		except ODEintegrationError as emg:
			print(emg, file=sys.stderr)
			print('pdic = %s'%repr(self.params.pardict), file=sys.stderr)
			return NegInf
		if math.isnan( lnprob ):
			# IF we get a NaN...
			# We should NOT ignore this since passing -inf is equivalent to
			#   forbidding this parameter set value. So this is an additional
			#   constraint we place on our parameters that we MUST be told
			#   about. When you encounter this warning, investigate!
			print('** WARNING! lnprob encountered NaN (and returned -inf) for:\npvec = %s'%repr(self.params.pardict), file=sys.stderr)
			return NegInf
		return lnprob


def solve_ivp(odefunc,t,y0,method='BDF', dense_output=False, events=None, vectorized=False, **options):
	import scipy.integrate
	try:
		res= scipy.integrate.solve_ivp(odefunc,(t[0],t[-1]),y0,method=method,t_eval=t,dense_output=dense_output,events=events,vectorized=vectorized,**options)
	except:
		raise ODEintegrationError('** WARNING! solve_ivp raised an exception.')
	if res.success:
		return res.y.T
	raise ODEintegrationError('** WARNING! solve_ivp integration Error:\n %s'%repr(res))


def odeint(*args,**kwargs):
	import scipy.integrate
	kwargs['full_output'] = True
	kwargs['printmessg'] = False
	kwargs['mxhnil'] = 1
	res,moreouts = scipy.integrate.odeint(*args,**kwargs)
	if moreouts['message'] == 'Integration successful.':
		return res
	raise ODEintegrationError('** WARNING! ODE integration Error:\n  \"%s\"'%moreouts['message'])
