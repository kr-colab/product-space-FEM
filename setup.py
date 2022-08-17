from codecs import open as codecs_open
from setuptools import setup, find_packages
from warnings import warn
import os


# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# After exec'ing this file we have psf_version defined.
psf_version = None  # Keep PEP8 happy.
version_file = os.path.join("product_fem", "_version.py")
with open(version_file) as f:
    exec(f.read())

setup(name='product_fem',
      version=psf_version,
      description=u"Solve equations on product function spaces.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[],
      keywords=['finite elements', 'population genetics'],
      author=u"Matt Lukac",
      author_email='mlukac@uoregon.edu',
      url='https://github.com/mattlukac/product-space-FEM',
      license='MIT',
      packages=['product_fem'],
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'fenics',
          'matplotlib',
          'numpy',
          'scipy',
      ],
      extras_require={
          'dev': [],
      },

      setup_requires=[],
      project_urls={
          'Bug Reports': 'https://github.com/mattlukac/product-space-FEM/issues',
          'Source': 'https://github.com/mattlukac/product-space-FEM',
      },
)
