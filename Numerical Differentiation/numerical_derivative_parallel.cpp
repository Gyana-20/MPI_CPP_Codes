// mpic++ -o run12 10numerical_derivative_parallel.cpp
// mpirun -np 4 ./run12

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <fstream>

double f(double x) {
    return x + pow(sin(x), 2);
}

double f_prime(double x) {
    return 1 + 2 * sin(x) * cos(x);
}

int main(int argc, char** argv) {
    int np = 0, rank = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
    double* x = nullptr;
    double a = 0.0, b = 0.0, h = 0.0;
    int num_points = 0;
    MPI_Status status;
    double* df = nullptr;
    std::ofstream fptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        num_points = 10000;  
        a = -1.0;  
        b = 1.0;   
        h = (b - a) / double(num_points - 1); 

        x = new double[num_points];
        x[0] = a;
        x[num_points - 1] = b;

        for (int i = 1; i < num_points - 1; i++) {
            x[i] = a + i * h;
        }

        df = new double [num_points]; 
    }

    // Broadcast common data to all processes
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&num_points, 1, MPI_INT, 0, comm);
    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, comm);

    // Calculate the number of points for each process
    int lnpt = (num_points / np) + (rank < (num_points % np) ? 1 : 0);

    if (lnpt <= 0) {
        std::cerr << "Invalid lnpt for rank " << rank << std::endl;
        MPI_Finalize();
        return -1;
    }

    // Allocate space for local points and ghost points (2 ghost points on each side)
    double* lx = new double[lnpt + 4]; // +4 for 2 ghost points on each side

    // Set up scatter counts and displacements for Scatterv
    int* send_cnt = new int[np];
    int* send_dis = new int[np];

    for (int i = 0; i < np; i++) {
        send_cnt[i] = (num_points / np) + (i < (num_points % np) ? 1 : 0);
        send_dis[i] = (i == 0) ? 0 : send_dis[i - 1] + send_cnt[i - 1];
    }

    // Scatter points to each process
    MPI_Scatterv(x, send_cnt, send_dis, MPI_DOUBLE, lx + 2, lnpt, MPI_DOUBLE, 0, comm);

    // Exchange two ghost points between neighboring processes
    if (rank != 0) {
        MPI_Send(&lx[2], 2, MPI_DOUBLE, rank - 1, 100, comm); // send first two local points to the left neighbor
        MPI_Recv(&lx[0], 2, MPI_DOUBLE, rank - 1, 100, comm, &status); // receive two ghost points from the left neighbor
    }

    if (rank != np - 1) {
        MPI_Send(&lx[lnpt], 2, MPI_DOUBLE, rank + 1, 100, comm); // send last two local points to the right neighbor
        MPI_Recv(&lx[lnpt + 2], 2, MPI_DOUBLE, rank + 1, 100, comm, &status); // receive two ghost points from the right neighbor
    }

    // Allocate space for local derivatives
    double* ldf = new double[lnpt];

    // Compute local derivatives
    if (rank == 0) {
        // First process: forward difference for the first point, central difference for the rest
        for (int i = 0; i < lnpt; i++) {
            if (i == 0 || i==1) {
                // Forward difference for the very first point
                ldf[i] = (-3 * f(lx[2]) + 4 * f(lx[3]) - f(lx[4])) / (2 * h);
            } else if (i == lnpt - 1 && np == 1) {
                // If only one process, backward difference for the last point
                ldf[i] = (3 * f(lx[lnpt + 1]) - 4 * f(lx[lnpt]) + f(lx[lnpt - 1])) / (2 * h);
            } else {
                // Central difference for the other points
                ldf[i] = (-f(lx[i + 4]) + 8 * f(lx[i + 3]) - 8 * f(lx[i + 1]) + f(lx[i])) / (12 * h);
            }
        }
    } else if (rank == np - 1) {
        // Last process: backward difference for the last point, central difference for the rest
        for (int i = 0; i < lnpt; i++) {
            if (i == lnpt - 1 || i==lnpt-2) {
                // Backward difference for the last point
                ldf[i] = (3 * f(lx[lnpt + 1]) - 4 * f(lx[lnpt]) + f(lx[lnpt - 1])) / (2 * h);
            } else {
                // Central difference for the other points
                ldf[i] = (-f(lx[i + 4]) + 8 * f(lx[i + 3]) - 8 * f(lx[i + 1]) + f(lx[i])) / (12 * h);
            }
        }
    } else {
        // Other processes: use central difference for all points
        for (int i = 0; i < lnpt; i++) {
            ldf[i] = (-f(lx[i + 4]) + 8 * f(lx[i + 3]) - 8 * f(lx[i + 1]) + f(lx[i])) / (12 * h);
        }
    }

    // Gather all local derivatives to the root process
    MPI_Gatherv(ldf, lnpt, MPI_DOUBLE, df, send_cnt, send_dis, MPI_DOUBLE, 0, comm);

    // Write the result to a file in the root process
    if (rank == 0) {
 if (rank == 0) {
    std::cout << "Writing to file..." << std::endl;
    fptr.open("10plot_parallel_data.txt");
    for (int i = 0; i < num_points; i++) {
        fptr << x[i] << " " << f_prime(x[i]) << " " << df[i] << std::endl;
    }
    fptr.close();
    std::cout << "File writing complete." << std::endl;
}

        delete[] x;
        delete[] df;
    }

    delete[] lx;
    delete[] send_cnt;
    delete[] send_dis;
    delete[] ldf;

    MPI_Finalize();
    return 0;
}
