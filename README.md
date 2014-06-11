Instructions for use
====================

The C code which computes the first- and second-order moments is in the
threelocus/ directory. To build this code and the accompanying Python
interface, execute the following two commands.

1. cd threelocus/
2. make python

This code requires Cython, Python and Google sparsehash
(https://code.google.com/p/sparsehash/) to compile. Additionally, you
need a C++11 compliant compiler like GCC 4.8.

Example code which makes use of this module is in code/. The file
estimation.py performs E&R simulations and then estimates the selection
parameter. See also the associated library modules in code/lib.
