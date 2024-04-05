.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10932778.svg
  :target: https://doi.org/10.5281/zenodo.10932778

Introduction
============

``pyRASP`` is a Python package written in support of the "RASP" work from the lab of Steven F. Lee. The paper relating to this code can be found at `The Journal of Physical Chemistry B <https://doi.org/10.1021/acs.jpcb.4c00174>`_. The code is a set of python classes that can be run from scripts, interactive notebooks and so on to analyse microscopy data. An example notebook is provided, showing user analyses. The code has been tested in Python 3.10.12.

A list of package requirements are noted in the "requirements.txt" file. These can be installed with the command:

``pip install -r requirements.txt``

The current implementation has been developed in Python 3 and tested on an Ubuntu 20.04 machine, as well as several Windows 10 machines.

Limitations
***********

The RASP concept assumes that you are imaging something that is tissue/cell like, *i.e.* has autofluorescence. If you run your radiality calibration on data that do not contain autofluorescence, likely the code will pick out genuinely fluorecent spots (if you have any) and so remove real stuff from your data.

Getting Started with pyRASP
***************************

