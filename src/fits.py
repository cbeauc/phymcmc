import numpy
import scipy.optimize
PosInf = float('+inf')

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


def rcost(pvec, model, params, args):
	return numpy.ones(pvec.shape)*scost(pvec, model, params, args)

def scost(pvec, model, params, args):
	pvec = 10.0**pvec
	try:
		tmp = model(pvec,params)
	except ValueError: # Problem with parameters
		return PosInf
	#print( params.pardict )
	try:
		ssr = tmp.get_ssr(args)
		#print('ssr = '+repr(ssr)+'\n')
		return ssr
	except: # Unknown/unforeseen problem
		print('WARNING: Your code believes the parameters are valid but the call to get_ssr failed. Figure out why and fix this problem.')
		return PosInf


def perform_fit(model, params, args, verbose=True, rep_fit=3):
	if verbose:
		print(params.parfit)
		print(params.vector)

	# It's best to fit parameters in log space
	pvec = numpy.log10(params.vector)

	# Do rep_fit fits w leastsq, a wrapper of MINPACK's Levenberg-Marquardt
	for rep in range(rep_fit):
		lsout = scipy.optimize.leastsq(rcost, pvec, args=(model,params,args), maxfev=7200, full_output=True)[0:3]
		pvec = lsout[0]
		params.vector = 10.0**pvec
		ssr = lsout[2]['fvec'][0]
		if verbose:
			print( 'Levenberg-Marquardt, rep %d (ssr = %g)' % (rep,ssr) )
			print( params.pardict )

	# One long but more accurate fit using the Nelder-Mead downhill simplex
	[pvec,ssr] = scipy.optimize.fmin(scost, pvec, args=(model,params,args), full_output=True, disp=False)[0:2]
	params.vector = 10.0**pvec
	if verbose:
		print( 'Nelder-Mead (ssr = %g)' % ssr )
		print( params.pardict )

	# One last fit w leastsq, a wrapper of MINPACK's Levenberg-Marquardt
	lsout = scipy.optimize.leastsq(rcost, pvec, args=(model,params,args), maxfev=7200, full_output=True)
	pvec = lsout[0]
	params.vector = 10.0**pvec
	ssr = lsout[2]['fvec'][0]
	if verbose:
		print( 'Levenberg-Marquardt (final fit, ssr = %g)' % ssr )
		print( params.pardict )

	# Returns (best-fit parameters, SSR)
	return (params, ssr)

