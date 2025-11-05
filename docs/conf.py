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

# Mock imports for packages that might not be available or require system dependencies
autodoc_mock_imports = [
    "cdo", 
    "earthengine-api", 
    "geemap", 
    "skimage", 
    "pyTMD", 
    "pytmd",
    "utide",
    "folium",
    "pymcdm",
    "configobj"
]

project = "environmentaltools"
author = "Manuel Cobos"
release = "2026.0.1"

html_theme = "sphinx_rtd_theme"

