import os

import pkg_resources
from setuptools import find_packages, setup
 
setup(
  name = "sdscripts", 
  packages = find_packages(),
  version= "0.0.1",
  install_requires=[
    str(r)
    for r in pkg_resources.parse_requirements(
      open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    )
  ],
)