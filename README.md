Active Virtual Screening
------------------------

MATLAB code to replicate experiments from

    Garnett, R., Gärtner, T. Vogt, M. and Bajorath, J.
	Introducing the ‘active search’ method for iterative virtual screening.
	Journal of Computer-Aided Molecular Design, 29(4):305–314.
    DOI: 10.1007/s10822-015-9832-9

Dependencies
------------

* [Active learning toolbox](https://github.com/rmgarnett/active_learning)
* [Active search toolbox](https://github.com/rmgarnett/active_search)

Obtaining Data
--------------

Raw data can be downloaded as a compressed tar file
[here](https://www.dropbox.com/s/u0tztvhozjpmmi9/processed_data.tar.xz).


Preparing Data
--------------

You can precompute nearest neighbors with
`calculate_nearest_neighbors`. You will need to compile the MEX file
`jaccard_nn.c` first:

    mex jaccard_nn.c

Running Experiments
-------------------

You can replicate the active search experiments using
`run_experiments`.
