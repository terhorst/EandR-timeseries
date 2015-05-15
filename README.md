This archive contains code to perform the likelihood calculations
described in Terhorst, Schlötterer & Song (2015). Those calculations
rely on the Python library `threelocus` which is also included. In
order to execute this code, first build the library on your machine as
detailed in the file `threelocus/README.txt`.

The file `example.py` is example of how to use this library to perform
inference. It analyzes a test data set (also included) in order to test
for and estimate selection. Refer to the comments in that
file for further details. Execute `example.py` using the wrapper script
`example.sh` in order to ensure that the threelocus library can be
located on your system's library path.
