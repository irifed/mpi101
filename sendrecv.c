#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#define NREPS 10
#define ROOT_RANK 0

#define KILO (1024)
#define MEGA (KILO*KILO)

void print_matrix(const int nrows, const int ncols, const double *matrix) {
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            printf("%10f ", matrix[row*ncols + col]);
        }
        printf("\n");
    }
}

double time_sendrecv(const int msg_size, const int src, const int dst) {
   
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // allocate buffer for `msg_size` bytes
    char *buf = (char *) malloc(msg_size);
    assert(buf != NULL);

    // leave `buf` unfilled because we do not need the contents now

    double t1 = 0.0;
    double t2 = 0.0;

    // `tag` can be arbitrary in this example
    const int tag = 0;

    t1 = MPI_Wtime();
    for (int rep = 0; rep < NREPS; rep++) {

        if (my_rank == dst) {
            MPI_Status status;

            MPI_Recv(buf, msg_size, MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);
        } else if (my_rank == src) {
            MPI_Send(buf, msg_size, MPI_BYTE, dst, tag, MPI_COMM_WORLD);
        } else {
            // yay! nothing to do
        }

    }
    t2 = MPI_Wtime();
    double etime = (t2 - t1)/((double) NREPS);

    // find out max etime across all ranks and store it on ROOT_RANK
    double max_etime;
    MPI_Reduce(&etime, &max_etime, 1, MPI_DOUBLE, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);

    free(buf);

    return max_etime;
}

int main(int argc, char *argv[]) {
    
    MPI_Init(&argc, &argv);

    int nprocs, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    const long msg_sizes[] = {1, 100, KILO, MEGA, 10*MEGA};
    const int n_msg_sizes = sizeof(msg_sizes)/sizeof(long);

    double *time_matrix = (double *) calloc(nprocs * nprocs, sizeof(double));
    assert(time_matrix != NULL);

    for (int i = 0; i < n_msg_sizes; i++) {

        for (int src_rank = 0; src_rank < nprocs; src_rank++) {
            for (int dst_rank = 0; dst_rank < nprocs; dst_rank++) {
                double etime = 0.0; 

                if (dst_rank == src_rank) continue;

                etime = time_sendrecv(msg_sizes[i], src_rank, dst_rank);
                if (my_rank == ROOT_RANK) {
                    time_matrix[src_rank*nprocs + dst_rank] = etime;
                }

            }
        }

        if (my_rank == ROOT_RANK) {
            printf("msg_size = %ld\n", msg_sizes[i]);
            print_matrix(nprocs, nprocs, time_matrix);
        }
    }

    free(time_matrix);
    MPI_Finalize();
    return 0;
}
