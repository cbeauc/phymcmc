from distutils.core import setup

setup(
	name = "phymbie",
	version = "0.1",
	author = "Catherine Beauchemin",
	author_email = "cbeau@users.sourceforge.net",
	description = "The phymbie fitting/mcmc library.",
	url = "http://phymbie.physics.ryerson.ca",
	license = "See file LICENSE",
	packages = [
		"phymbie",
		"phymbie.emcee"
	],
	package_dir = {'phymbie':'src'}
)
