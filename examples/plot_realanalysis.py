#!/usr/bin/env python
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

import phymbie.plot
import phymbie.model

strains = ['H275', 'Y275', 'I223', 'V223']
sdir = '/tmp/Fits/'
chain_files = [sdir+strn+'/'+strn+'_chain.hdf5' for strn in strains]

parlabeldict = {
	'ssr': r'Sum of squared residuals',
	'cpfu': r'Inf.\ clearance, $c_\mathrm{pfu}$ (1/h)',
	'tE': r'Eclipse phase, $\tau_E$ (h)',
	'tI': r'Infectious lifespan, $\tau_I$ (h)',
	'b': r'Infec.\ rate, $\beta$, (mL/PFU/h)',
	'vom': r'Inoculum, $V_{0,MC}$, (PFU/mL)',
	'pfm': r'Prod.\ rate MC (PFU/mL/h)',
	'pfs': r'Prod.\ rate SC (PFU/mL/h)',
	'pr': r'Prod.\ rate (RNA/mL/h)',
	'p2r': r'Inoculum ratio, (RNA/PFU)',
	'R0': r'Basic repro.\ num., $R_0$',
	'tinf': r'Infecting time, $t_\mathrm{inf}$ (min)',
	'prcell': r'Prod.\ rate (RNA/cell/h)',
	'pfcell': r'Prod.\ rate (PFU/cell/h)',
	'burst': r'RNA burst size (RNA/cell)',
	'inf2rna': r'$R_0$/burst (infection/RNA)',
	'pfs2fm': r'PFU prod.\ ratio, $p_\mathrm{SC}/p_\mathrm{MC}$',
	'pf2r': r'Infectiousness, $p_\mathrm{PFU/RNA}$',
	'moi': r'MOI',
	'vop2r': r'Inocul.\ ratio, (PFU/RNA)'
}

# Making the triangle plots
if False:
	plotpars = ['ssr','cpfu','tE','tI','b','vom','pfm','pfs','pr','p2r']
	for si,strain in enumerate(strains):
		plotparlabels = plotpars
		fig = phymbie.plot.triangle(plotpars, plotparlabels, chain_files[si])
		fig.savefig('realouts/'+strain+'_triangle.png')

# Making a grid of relative hist plots
if False:
	strains = ['H275', 'Y275', 'I223', 'V223']
	chain_files = [sdir+strn+'/'+strn+'_chain.hdf5' for strn in strains]
	relative = [0, 0, 2, 2]
	colors = ['black','red','blue', 'green']
	plotpars = ['tE', 'tinf', 'tI', 'cpfu', 'prcell', 'pfcell', 'pf2r', 'pfs2fm', 'b', 'vom', 'vop2r']
	plotlabels = [parlabeldict[key] for key in plotpars]
	fig = phymbie.plot.hist_grid(plotpars, chain_files, colors, dims=(3,4), labels=plotlabels, relative=relative)
	fig.savefig('realouts/hist_rel_HYIV.pdf', bbox_inches='tight')

# Making a grid of absolute hist plots for I vs V
if False:
	strains = ['I223', 'V223']
	chain_files = [sdir+strn+'/'+strn+'_chain.hdf5' for strn in strains]
	colors = ['blue', 'green']
	relative = []
	plotpars = ['tE','pfcell','tinf','prcell','tI','b']
	plotlabels = [parlabeldict[key] for key in plotpars]
	fig = phymbie.plot.hist_grid(plotpars, chain_files, colors, dims=(3,2), labels=plotlabels, relative=relative)
	fig.savefig('realouts/hist_IV.pdf', bbox_inches='tight')

# Making a grid of absolute hist plots for H vs I
if True:
	strains = ['H275', 'Y275', 'I223', 'V223']
	chain_files = [sdir+strn+'/'+strn+'_chain.hdf5' for strn in strains]
	colors = ['black','red','blue', 'green']
	relative = []
	plotpars = ['cpfu','pfs2fm','pf2r']
	plotlabels = [parlabeldict[key] for key in plotpars]
	fig = phymbie.plot.hist_grid(plotpars, chain_files, colors, dims=(1,3), labels=plotlabels, relative=relative)
	fig.savefig('realouts/hist_abs_HI.pdf', bbox_inches='tight')

