import os
import sys
sys.path.insert(0, os.path.abspath("../src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # si usas docstrings estilo Google o NumPy
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",    # opcional: genera resúmenes automáticos
]

autosummary_generate = True

project = "environmentaltools"
author = "Manuel Cobos"
release = "2026.0.1"

html_theme = "sphinx_rtd_theme"

