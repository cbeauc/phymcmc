# Copyright (C) 2014-2019 Catherine Beauchemin <cbeau@users.sourceforge.net>
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
		if min(self.pardict.values()) < 0.0:
			raise ValueError(repr(self.pardict))
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
		self.d = self.pdic['nI']/self.pdic['tI']
		self.y0 = numpy.hstack((self.pdic['V0'], self.pdic['N'], numpy.zeros(self.pdic['nI'])))
		res = phymcmc.solve_ivp(self.derivative,t,self.y0)
		# Replace data points below Vlim by Vlim
		res[:,0] = numpy.maximum(self.pdic['Vlim'],res[:,0])
		return numpy.hstack((res[:,:2],numpy.sum(res[:,2:],axis=1,keepdims=True)))

	def get_normalized_ssr(self,pvec):
		""" Computes total normalized SSR, i.e. SSR/stdev. """
		try:
			self.params.vector = pvec
		except ValueError:
			return float('inf')
		dathpi,datV,sigV = self.data
		# Required to tell solve_ivp that initial conditions (y0) are for t=0
		tsim = numpy.hstack((0.0,dathpi))
		#	now need to throw out first row (t=0) to get rid of t=0 results
		residuals = numpy.log10( self.get_solution(tsim)[1:,0]/datV )/sigV
		return (residuals**2.0).sum()
