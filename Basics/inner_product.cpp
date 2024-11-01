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

- All the process print their stored results.(commented)
  Process 0 prints the results.
  
Author: GYANA RANJAN NAYAK

Usage:
mpic++ -o {exe} {filename.cpp}
mpirun -np {nprocs} ./{exe}
*/


# include <iostream>
# include <random>
# include <iomanip>
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

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"Enter the dimension of the vectors: ";
        std::cin>>dim;

        if (dim <=0){
            std::cerr<<"Error: dimension must be positive."<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

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


    // uncomment the below to see the values stored in each process
    // every process will have the same value
    // I am printining it on process 0;
    // std::cout<<"Rank "<<rank<<" || value:"<<global_inner_prod<<std::endl; 

    double time_1 = MPI_Wtime();

    if (rank == 0){
        std::cout<<std::fixed<<std::setprecision(10);
        std::cout<<"Inner Product value: "<<global_inner_prod<<std::endl;
        std::cout<<"Time elapsed: "<<time_1 - time_0 <<" sec"<<std::endl;
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
