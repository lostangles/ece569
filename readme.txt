To compile:
cd build
cmake ..
make

To run:
cd build
./runScript

Time stamped outputs will be printed to the console and saved in the *.out files in the runScript directory

All libraries are self contained - using the stb_*.h single file include headers.

The output test images reside within the same directory as the runScript in the format
of test#.jpg

The baseline project (before modifying the kernels) is located in the folder baseline and can be run/executed using the same instructions as above
