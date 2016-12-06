[![Build Status](https://travis-ci.org/TorbjornT/pyAccuRT.svg?branch=master)](https://travis-ci.org/TorbjornT/pyAccuRT)  (tests are rudimentary at the moment)

``ReadART`` class is for

 - reading output from AccuRT radiative transfer model
 - creating simple plots
 - calculating albedo, transmittance, some other parameters
 - writing data to Matlab- or NetCDF-file.

Dependencies:

 - ``python`` 3.x (``python`` 2 is *not* supported)
 - ``numpy``
 - ``matplotlib``
 - ``scipy``
    - ``scipy.io`` for writing Matlab and NetCDF files
    - ``scipy.ndimage.filters.gaussian_filter1d`` for gaussian smoothing


Documentation is limited to non-existent, see an example in the examples-folder.

# Installation

Quite standard, clone/download repo and run

    python setup.py install




AccuRT is a registered trademark of Geminor Inc. in the United States.

The code in this repository is licensed under the MIT License.
