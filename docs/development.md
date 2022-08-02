---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{currentmodule} product_fem
```

(sec_documentation)=

These are notes for working on the project itself.

# Documentation

The documentation is built with
[jupyterbook](https://jupyterbook.org/en/stable/intro.html) via
[jupytext](https://jupytext.readthedocs.io/en/latest),
in the [myst format](https://jupyterbook.org/en/stable/content/myst.html),
which allows for calling of sphinx directives.
To build the documentation, run `make` in this directory
and navigate a browser to `_build/html/`.
