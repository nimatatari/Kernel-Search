# Kernel-Search

A-star_Base_Search
(A-star search of base functions in an ML implementation)
Written by N. Tatari (n.tatari@wustl.edu)
Washington University in St. Louis, 2018-2019
______________________________________________________________________



The code is based on the search method (A-star) over a randomly 
distributed set of functions, that are used as the base functions 
(or base vectors), to train an ML algorithm.

Finding a data-specific ML model, customized for the data set
with the aim to enhance the pattern recognition process is central
to the algorithm.

In the data.py file, simple data sets can be generated, based on 
known mathematical functions. The code implemented in baseSearch.py
is an algorithm that searches for the best set of base functions that
can be trained by that specific data.

The functions are generated randomly at each branching point of the
search tree (e.g. tan(), tanh(), exp(), cosine() etc. ), and the
function parameters are optimized. Finally, the training of the model
is performed by gradient descent.
