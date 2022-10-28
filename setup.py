from setuptools import setup
import os

# After exec'ing this file we have psf_version defined.
psf_version = None  # Keep PEP8 happy.
version_file = os.path.join("product_fem", "_version.py")
with open(version_file) as f:
    exec(f.read())

setup(
    # The package name along with all the other metadata is specified in setup.cfg
    # However, GitHub's dependency graph can't see the package unless we put this here.
    name='product_fem',
    version=psf_version,
)
