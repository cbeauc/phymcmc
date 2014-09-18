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
# Useful phymbie/virus-specific modelling tidbits
#

import scipy.integrate
def odeint(*args,**kwargs):
	assert 'mxstep' not in kwargs
	kwargs['mxstep'] = 4000000
	return scipy.integrate.odeint(*args,**kwargs)


# Useful dictionaries 
parnames = {} # parameter names
parsymbs = {} # symbols (units)
# Base parameters
parnames['tI'] = r'Infectious phase'
parsymbs['tI'] = r'$\tau_I$ (h)'
parnames['tE'] = r'Eclipse phase'
parsymbs['tE'] = r'$\tau_E$ (h)'
parnames['cpfu'] = r'Rate infec.\ loss'
parsymbs['cpfu'] = r'$c$ (h$^{-1}$)'
parnames['crna'] = r'Rate infec.\ loss'
parsymbs['crna'] = r'$c$ (h$^{-1}$)'
parnames['b'] = r'Infect.\ rate'
parsymbs['b'] = r'$\beta$ (mL/pfu/h)'
parnames['pfs'] = r'Prod.\ rate'
parsymbs['pfs'] = r'$p_\mathrm{pfu,SC}$ (pfu/mL/h)'
parnames['pfm'] = r'Prod.\ rate'
parsymbs['pfm'] = r'$p_\mathrm{pfu,MC}$ (pfu/mL/h)'
parnames['pr'] = r'Prod.\ rate'
parsymbs['pr'] = r'$p_\mathrm{rnac}$ (RNA/mL/h)'
# Biological params
parnames['prod'] = r'Prod.\ rate'
parsymbs['prod'] = r'$p_\mathrm{RNA}$ (RNA/h/cell)'
parnames['R0'] = r'Basic repro.\ num.'
parsymbs['R0'] = r'$R_0$'
parnames['burst'] = r'Burst size'
parsymbs['burst'] = r'$B$ (RNA/cell)'
parnames['tinf'] = r'Infecting time'
parsymbs['tinf'] = r'$t_\mathrm{inf}$ (min)'
parnames['inf2rna'] = r'Infectiousness'
parsymbs['inf2rna'] = r'$I_\mathrm{RNA}$ (infection/RNA)'
parnames['pfs2fm'] = r'DIP effect'
parsymbs['pfs2fm'] = r'$p_\mathrm{pfu,SC}$/$p_\mathrm{pfu,MC}$'
parnames['pf2r'] = r'Prod ratio'
parsymbs['pf2r'] = r'$p_\mathrm{pfu}$/$p_\mathrm{RNA}$'
# Internal params
parnames['vom'] = r'Inoculum'
parsymbs['vom'] = r'$V_{0,\mathrm{MC}}$ (pfu/mL)'
parnames['p2r'] = r'Inoc.\ ratio'
parsymbs['p2r'] = r'$V_{0,\mathrm{pfu}}/V_{0,\mathrm{rna}}$ (PFU/RNA)'

