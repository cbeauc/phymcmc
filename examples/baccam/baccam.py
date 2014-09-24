import numpy
import phymcmc.model
NegInf = float('-inf')

class model(object):
	def __init__(self, pvec, par):
		par.vector = pvec
		# Test params validity and complain if error
		if not params_are_valid( par.pardict ):
			raise ValueError(par.pardict)
		self.pdic = par.pardict
		self.d = self.pdic['nI']/self.pdic['tI']
		self.y0 = numpy.hstack((self.pdic['V0'], self.pdic['N'], numpy.zeros(self.pdic['nI'])))

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
		res = phymcmc.model.odeint(self.derivative,self.y0,numpy.hstack((0.0,t)))[1:,:]
		return numpy.hstack((res[:,0:2],numpy.mean(res[:,2:],axis=1,keepdims=True)))

	def get_residuals(self,(dathpi,datV)):
		""" Computes the residuals between the model and the data. """
		# Replace data points below Vlim by Vlim
		mV = numpy.maximum(self.pdic['Vlim'],self.get_solution(dathpi)[:,0])
		return numpy.log10( mV/datV )

	def get_ssr(self,data):
		""" Computes total SSR from residuals. """
		return (self.get_residuals(data)**2.0).sum()

	@classmethod
	def get_lnprob(cls,pvec,par,dat):
		"""
			Determine the lnprob for the model, given the parameters.
			This function MUST be defined in order for phymcmc.mcmc to work.
			In this function, you can do additional calculations like add
			some correction to the log posterior likelihood function (e.g.
			for running a log parameter in lin scale or whatever). But
			DO NOT check here for NaN. The sanity parsing of the SSR value
			is done by the lnprobfn function in the phymcmc MCMC library.
		"""
		try:
			self = cls(pvec,par)
		except ValueError:
			return NegInf
		return -self.get_ssr(dat)


def params_are_valid(pdic):
	""" Assess validity of model parameters. """
	if min(pdic.values()) < 0.0:
		return False
	if not (0.01 < pdic['c'] < 10.0):
		return False
	return True

