# HPC_Project
sustech HPC
cd implicit/explicit

If you want to run the functional code containing hdf5, make explicit.out/implicit.out directly. bsub<my-script.

If you want to see the performance of the program, you can go to the no_output folder, where the hdf5 read, write and output are closed.

Execute command in basic format:mpirun -np 1 ./explicit.out -restart 0 -n 100
