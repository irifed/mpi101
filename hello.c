#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    // initialize MPI library
    MPI_Init(&argc, &argv);

    int nprocs, myrank;

    // MPI_COMM_WORLD is built-in communicator containing all procs
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    printf("Hello from proc %d of %d\n", myrank, nprocs);

    // cleanup
    MPI_Finalize();

    return 0;
}
