# Copyright (C) 2014-2020 Catherine Beauchemin <cbeau@users.sourceforge.net>
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
###############################################################################

from setuptools import setup

setup(
	name = 'phymcmc',
	version = '0.5',
	author = 'Catherine Beauchemin',
	author_email = 'cbeau@users.sourceforge.net',
	url = 'https://github.com/cbeauc/phymcmc',
	license = 'See LICENSE file',
	description = 'Wraps emcee providing convenient functions and scripts',
	long_description = open("README.rst").read(),
	package_data = {"": ["LICENSE"]},
	packages = [
		'phymcmc',
		'phymcmc.corner',
		'phymcmc.emcee',
		'phymcmc.emcee.backends',
		'phymcmc.emcee.moves',
	],
	scripts = [
		'bin/phymcmc_fix_fracaccept',
		'bin/phymcmc_parstats',
		'bin/phymcmc_peakachain',
		'bin/phymcmc_plotter',
		'bin/phymcmc_rmderived'
	],
	package_dir = {'phymcmc':'lib'}
)
