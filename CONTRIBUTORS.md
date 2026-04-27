# Contributors

This file acknowledges all individuals and organizations who have contributed to this project.  
Contributions include code, documentation, design, testing, project management, and community support.

---

## How to be added
If you have contributed to this project, please submit a pull request adding your name to the list below.  
Please include your GitHub username and a short description of your contribution.

---

## Check list
1. Update the CHANGELOG.md for the upcoming version.
2. Update docstrings of functions and comments.
3. Update the package's init.py with the new functions.
4. Delete the _build and _autosummary folders from docs to ensure nothing old is linked.
5. Check that the module's .rst document includes the new functions so that Sphinx autosummary can generate them.
6. Add any new libraries imported to the autodoc_mock_imports variable in docs/conf.py.
7. Update the version in conf.py and pyproject.toml according to CHANGELOG.md.
8. Run pip install -e . in your virtual environment where you are modifying the package - Successfully installed environmentaltools-x.x.x
9. Run sphinx-build -b html docs docs/_build and verify there are no errors or warnings. Modify and add functions or information required by Sphinx until it compiles.
10. Run python -m build
11. Run git tag x.x.x
12. Run git push origin x.x.x
13. When the changes are significant, add them as a release on Github.

---

## Core Team
- **Manuel Cobos** – Project lead, architecture, and core development  
  [@mcobosb](https://github.com/mcobosb) - Environmental Fluid Dynamics Groug (https://dinamicambiental.es),  Andalusian Institute for Earth System Research (IISTA), Department of Structural Engineering and Fluid Mechanics, University of Granada.

- **Pedro Otiñar** – Project development  
  [@potinar](https://github.com/potinar) - Environmental Fluid Dynamics Groug (https://dinamicambiental.es),  Andalusian Institute for Earth System Research (IISTA), Department of Structural Engineering and Fluid Mechanics, University of Granada.

---

## Community Contributors
- **John Smith** – Code optimization and testing  
  [@johnsmith](https://github.com/johnsmith)

---

## Acknowledgements
We thank the following organizations and communities for their support:
- University of Granada  
- Open-source geospatial community  
- All contributors who have shared ideas, feedback, and improvements

---

## Additional Resources
- See also: [GitHub Contributors Graph](../../graphs/contributors)