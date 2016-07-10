
class ParamStruct(object):
	def __init__(self,pardict,parfit):
		self.pardict = pardict
		self.parfit = parfit
	@property
	def vector(self):
		return tuple(self.pardict[key] for key in self.parfit)
	@vector.setter
	def vector(self,fittedpars):
		for key,val in zip(self.parfit,fittedpars):
			self.pardict[key] = val

