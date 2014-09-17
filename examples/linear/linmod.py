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
#                                   Utilities
#
# =============================================================================
#

NegInf = float('-inf')

class model(object):
	def __init__(self, pvec, par):
		par.vector = pvec
		# Test params validity and complain if error
		if not params_are_valid( par.pardict ):
			raise ValueError(par.pardict)
		self.par = par.pardict
	def get_solution(self,data):
		""" Solve the model and obtain the model-generated data prediction. """
		return self.par['slope']*data[:,0]+self.par['yint']
	def get_residuals(self,data):
		""" Computes the residuals between the model and the data. """
		return data[:,1] - self.get_solution(data)
	def get_ssr(self,data):
		""" Computes total SSR from residuals. """
		return (self.get_residuals(data)**2.0).sum()
	@classmethod
	def get_lnprob(cls,pvec,par,dat):
		"""
			Determine the lnprob for the model, given the parameters.
			This function MUST be defined in order for phymbie.mcmc to work.
			In this function, you can do additional calculations like add
			some correction to the log posterior likelihood function (e.g.
			for running a log parameter in lin scale or whatever). But
			DO NOT check here for NaN. The sanity parsing of the SSR value
			is done by the lnprobfn function in the phymbie MCMC library.
		"""
		try:
			self = cls(pvec,par)
		except ValueError:
			return NegInf
		return -self.get_ssr(dat)

def params_are_valid(pdic):
	if min(pdic.values()) < 0.0:
		return False
	return True

