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
#
#                                   Utilities
#
# =============================================================================
#
import numpy
import phymcmc


class params(phymcmc.ParamStruct):
	def validate(self):
		""" Assess validity of model parameters. """
		if min(self.vector) < 0.0:
			raise ValueError(repr(self.pardict))


class line(phymcmc.base_model):
	def get_solution(self,x):
		""" Solve the model and obtain the model-generated data prediction. """
		return self.params.pardict['slope']*x+self.params.pardict['yint']
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
		self.par = self.params.pardict
		# Below, lnlikelihood's SSR is not divided by sigma^2 but should be
		# (see example baccam to see what I mean)
		# and we assume params priors are all uniform in linear space
		# 	i.e. prior = 1.0 -> ln[prior] = 0.0
		return -0.5*numpy.sum( (self.data[:,1] - self.get_solution(self.data[:,0]))**2.0 )
