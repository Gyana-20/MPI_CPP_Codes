/*
Program: 
This program distributes a NxN matrix 
between "nprocs" number of processes by using
column distribution. Here I have assumed that
N is divisible by nprocs.
MPI_Type_vector and MPI_Type_create_resized
routines are used.

Author:
Gyana Ranjan Nayak

usage:
mpic++ -o <name_of_exe>  <name_of_cpp_file>
mpirun  -np <nprocs> ./<name_of_exe>
*/

# include <iostream>
# include <mpi.h>

int main (int argc, char** argv)
{
    int nproc, rank;
    int N;
    double* A = nullptr;
    
    MPI_Init (&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    if (rank == 0){
        std::cout<<"Enter Number of rows (cols)[must be divisible by number of processors]: "<<std::endl;
        std::cin >> N;

        std::cout<<"\n";

        if (N%nproc != 0){
            std::cerr<<"Error: Number of rows (cols) must be divisible by number of processes"<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A = new double [N*N];

        for (int i = 0; i < N; i++){
            for(int j=0; j<N; j++){
                A[i*N+j] = i*N+j;
            }
        }

        std::cout<<"Initial Matrix: "<<std::endl;
        for(int i=0; i< N; i++){
            for (int j=0; j<N; j++){
                std::cout<<A[i*N+j]<<" ";
            }
            std::cout<<std::endl;
        }
    }

    MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = N/nproc;
    double* local_A = new double [N*local_n];

    MPI_Datatype columnType;


    MPI_Type_vector (N, local_n, N,  MPI_DOUBLE, &columnType);
    MPI_Type_create_resized(columnType,  0, local_n*sizeof(double), &columnType);
    MPI_Type_commit (&columnType);

    MPI_Scatter (A, 1, columnType, local_A, N*local_n, MPI_DOUBLE,0, MPI_COMM_WORLD);

    std::cout<<"Rank "<<rank<<" : "<<std::endl;

    for(int i=0; i< N; i++){
        for (int j=0; j<local_n; j++){
            std::cout<<local_A[i*local_n+j]<<" ";
        }
        std::cout<<std::endl;
    }

    if (rank == 0) delete[] A;
    delete[] local_A;

    MPI_Type_free(&columnType); 
    MPI_Finalize();

    return 0;
}