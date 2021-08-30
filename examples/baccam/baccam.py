# Copyright (C) 2014-2021 Catherine Beauchemin <cbeau@users.sourceforge.net>
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

import numpy
import phymcmc

class params(phymcmc.ParamStruct):
	def validate(self):
		""" Assess validity of model parameters. """
		if min(self.vector) < 0.0:
			raise ValueError(repr(self.pardict))
		# constrain param "c" to be in [0.01,10.0]/hour
		if not (0.01 < self.pardict['c'] < 10.0):
			raise ValueError(repr(self.pardict))


class model(phymcmc.base_model):
	def derivative(self,t,x):
		""" Return the derivative of each variable of the model. """
		(V,T,I) = (x[0],x[1],x[2:])
		dV = self.pdic['p']*numpy.sum(I) - self.pdic['c']*V
		dT = -self.pdic['b']*T*V
		dI1 = self.pdic['b']*T*V - self.d*I[0]
		dIi = -self.d*numpy.diff(I)
		return numpy.hstack((dV,dT,dI1,dIi))

	def get_solution(self,t):
		""" Solve the model and obtain the model-generated data prediction. """
		self.pdic = self.params.pardict
		# must be an integer
		self.pdic['nI'] = int(round(self.pdic['nI']))
		self.d = 1.0*self.pdic['nI']/self.pdic['tI']
		self.y0 = numpy.hstack((self.pdic['V0'], self.pdic['N'], numpy.zeros(self.pdic['nI'])))
		res = phymcmc.solve_ivp(self.derivative,t,self.y0)
		# Replace data points below Vlim by Vlim
		res[:,0] = numpy.maximum(self.pdic['Vlim'],res[:,0])
		# Return V, T, I=sum_i=1^nI I_i , dim=[len(t) rows, 3 columns]
		return numpy.hstack((res[:,:2],numpy.sum(res[:,2:],axis=1,keepdims=True)))

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
		# Handle explicitly disallowed params encountered
		try:
			self.params.vector = pvec
		except ValueError:
			return float('-inf')
		# Unpack the data we used to initialize the model instance
		dathpi,datV,sigV = self.data
		# Required to tell solve_ivp that initial conditions (y0) are for t=0
		# in case first data timepoint is not at t=0.
		tsim = numpy.hstack((0.0,dathpi))
		# Compute residuals in log10[ Vmodel / Vdata ]
		#	now need to throw out first row (t=0) to get rid of t=0 results
		residuals = numpy.log10( self.get_solution(tsim)[1:,0]/datV )/sigV
		# We assume a log-uniform prior for all estimated params (p, c, b, V0)
		lnprior = -numpy.log(numpy.prod(self.params.vector))
		# lnprob = lnlikeli + lnprior = -0.5 ssr/sig^2 + lnprior
		return -0.5*(residuals**2.0).sum() + lnprior
