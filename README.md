SAVSNet Python functions
========================

This folder contains the savsnet Python module, containing Python functions to analyse SAVSNet data.  The functions are generic, and other users may find them useful.


GP Smoothing of case timeseries
-------------------------------

The `savsnet/prev_ts_GP` module contains functions that build a Binomial regression model with a Gaussian process linear predictor to model a timeseries of Binomial random variables (e.g. prevalence).  See docstrings for each function for further details.  For convenience, the module may be run as a script, for example:
````bash
$ python savsnet/prev_ts_GP.py -c trauma -s dog cat -i 1000 -o pred myData.csv
````
where myData.csv is a CSV file containing (minimally) a 'Date' column (ISO format), 'Consult_reason', and 'Species' columns.

See 
````bash
$ python savsnet/prev_ts_GP.py --help
````
for further details.


Plotting of GP smooths
----------------------

The `savsnet/plot_ts_GP` module contains functions for plotting posterior predictive distributions from savsnet/prev_ts_GP output.  Documentation for individuals functions are contained in the docstrings.  For convenience, the module may be run as a script, for example:
````bash
$ python savsnet/plot_ts_GP.py -d myData.csv -s dog cat -c trauma -p pred_dog_trauma.pkl pred_cat_trauma.pkl -o gpFigure.pdf
````

See
````bash
$ python savsnet/plot_ts_GP.py --help 
````
for further details.


License
-------

This software is release under the MIT license.  Please refer to the LICENSE file contained in the same directory as this file for further details.
