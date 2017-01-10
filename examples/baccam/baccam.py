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
#
# =============================================================================

import numpy
import phymcmc

class model(phymcmc.base_model):

	def derivative(self,x,t):
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
		res = phymcmc.odeint(self.derivative,self.y0,numpy.hstack((0.0,t)))[1:,:]
		# Replace data points below Vlim by Vlim
		res[:,0] = numpy.maximum(self.pdic['Vlim'],res[:,0])
		return numpy.hstack((res[:,:2],numpy.mean(res[:,2:],axis=1,keepdims=True)))

	def get_normalized_ssr(self,pvec):
		""" Computes total normalized SSR, i.e. SSR/stdev. """
		self.params.vector = pvec
		# Test params validity and complain if error
		if not params_are_valid( self.params.pardict ):
			raise ValueError(self.params.pardict)
		dathpi,datV,sigV = self.data
		residuals = numpy.log10( self.get_solution(dathpi)[:,0]/datV )/sigV
		return (residuals**2.0).sum()


def params_are_valid(pdic):
	""" Assess validity of model parameters. """
	if min(pdic.values()) < 0.0:
		return False
	if not (0.01 < pdic['c'] < 10.0):
		return False
	return True
