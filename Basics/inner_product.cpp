/*
Program:
- Parallel Implementation of dot product

- Two random vectors of size "dim" (user input)
  are generated.(say a and b)

- a and b have been distributed among nprocs number of
  processes. Distribution is done using "MPI_Scatterv"
  routine. 

- Each process computes the dot product of the  portion
  of a and b it has received. And using  "MPI_Allreduce"
  the final result is  obtained and stored in all the 
  processes.

- All the process print their stored results.

Author: GYANA RANJAN NAYAK

Usage:
mpic++ -o {exe} {filename.cpp}
mpirun -np {nprocs} ./{exe}
*/


# include <iostream>
# include <random>
# include <mpi.h>

double inner_prod(double* a, double* b, int N){
    double sum = 0;

    for (int i=0; i<N; i++){
        sum += a[i]*b[i];
    }
    
    return sum;
}

void generate_vector (double* a, int N){
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_real_distribution<double> dist(0.0, 10.0);

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
    
    int dim, rank, nprocs;

    double* a = nullptr;
    double* b = nullptr;

    MPI_Init (&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0){
        std::cout<<"Enter the dimension of the vectors: ";
        std::cin>>dim;

        a = new double [dim];
        b = new double [dim];

        generate_vector(a, dim);
        generate_vector(b, dim);

        std::cout<<"a: ";
        print_v(a,dim);
        std::cout<<std::endl;
        std::cout<<"b: ";
        print_v(b,dim);
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

    double* local_a = new double[local_dim];
    double* local_b = new double [local_dim];

    MPI_Scatterv (&a[0], send_cnt, send_dis, MPI_DOUBLE,
                   local_a, local_dim, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    MPI_Scatterv (&b[0], send_cnt, send_dis, MPI_DOUBLE,
                   local_b, local_dim, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

    double local_inner_prod;
    local_inner_prod = inner_prod(local_a, local_b, local_dim);

    double global_inner_prod;
    
    MPI_Allreduce (&local_inner_prod, &global_inner_prod,
                  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::cout<<"Rank "<<rank<<" || value:"<<global_inner_prod<<std::endl;

    if (rank == 0){
        delete[] a;
        delete[] b;
    }

    delete[] local_a;
    delete[] local_b;
    delete[] send_cnt;
    delete[] send_dis;

    MPI_Finalize();
    return 0;
}