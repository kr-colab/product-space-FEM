
# Need to set PYTHONPATH so that we pick up the local module
PYPATH=${PWD}/..
PSF_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import product_fem; print(product_fem.__version__.split("+")[0])')

dev:
	PYTHONPATH=${PYPATH} jupyter-book build -W --keep-going .

dist:
	@echo Building distribution for product_fem version ${PSF_VERSION}
	sed -i s/__PSF_VERSION__/${PSF_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build
