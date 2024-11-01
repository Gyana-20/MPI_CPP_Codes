/*
Program:
- This program demonstrates the scattering of a vector x
  among "nprocs" number of processes using Scatterv routine
  and gathering them  back using Allgatherv routine.

Author: GYANA RANJAN NAYAK

Usage:
mpic++ -o {exe} {filename.cpp}  
mpirun -np {nprocs} ./exe
*/

# include <iostream>
# include <random>
# include <mpi.h>

void generate_vector (double* a, int N){
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_int_distribution<int> dist(0, 10);

    for (int i=0; i<N; i++){
        a[i] = dist(gen);
    }
}

void print_v (double* a, int N){
    for (int i=0; i<N; i++){
        std::cout << a[i] << " ";
    }
}

int main(int argc, char** argv){

    int nprocs, rank;
    int dim;
    double* x = nullptr;

    MPI_Init(&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    if (rank == 0){
        std::cout<<"Enter the dimension of the vectors: ";
        std::cin>>dim;

        if (dim <=0){
            std::cerr<<"Error: dimension must be positive."<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        x = new double [dim];
        generate_vector(x, dim); 
        
        std::cout<<"x: ";
        print_v(x,dim);
        std::cout<<std::endl;  
    }

MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_dim = (dim/nprocs) + (rank < dim%nprocs ? 1 : 0);

    int* send_cnt = new int [nprocs];
    int* send_dis = new int [nprocs];

    for (int i=0; i<nprocs; i++){
        send_cnt[i] = (dim/nprocs) + (i < dim%nprocs ? 1 : 0);

        if (i == 0) {send_dis[i] = 0;}
        else {
            send_dis[i] = send_dis[i-1] + send_cnt[i-1];
        }
    }

    double* local_x = new double[local_dim];

    MPI_Scatterv (&x[0], send_cnt, send_dis, MPI_DOUBLE,
                  local_x, local_dim, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);

    std::cout<<"Rank "<<rank <<" local_x: ";
    print_v(local_x, local_dim);
    std::cout<<std::endl;

    double* global_x = new double[dim];

    MPI_Allgatherv(local_x, local_dim, MPI_DOUBLE,
                    global_x, send_cnt, send_dis, MPI_DOUBLE,
                    MPI_COMM_WORLD);

    std::cout<<"Rank "<<rank <<" global_x: ";
    print_v(global_x, dim);
    std::cout<<std::endl;

    if (rank == 0){
        delete[] x;
    }
    
    delete[] local_x;
    delete[] send_cnt;
    delete[] send_dis;
    delete[] global_x;

    MPI_Finalize();
    return 0;
}
