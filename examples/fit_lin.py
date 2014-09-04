import phymbie
import phymbie.fits
import numpy

class mylinmodel(object):
	def __init__(self, pvec, par):
		par.vector = pvec
		self.par = par.pardict
	def get_solution(self,data):
		return self.par['slope']*data[:,0]+self.par['yint']
	def get_residuals(self,data):
		return data[:,1] - self.get_solution(data)
	def get_ssr(self,data):
		return (self.get_residuals(data)**2.0).sum()

datlin = numpy.loadtxt("lin.dat")

# Would either load params or initialize and fit
pdic = dict(slope=1.39,yint=573.13)
pfit = ['slope','yint']
params = phymbie.fits.ParamStruct(pdic,pfit)

# This is fitting of params if it had not yet been done
bfvec,bfssr = phymbie.fits.perform_fit(mylinmodel,params,datlin,verbose=True,rep_fit=3)
