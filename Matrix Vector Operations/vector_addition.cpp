/*
Program:
- This program demonstrates the implimentation of 
  Parallel vector addition using MPI_Scatterv and 
  MPI_Allgatherv routines.

Author: GYANA RANJAN NAYAK

Usage:
mpic++ -o {exe} {filename.cpp}  
mpirun -np {nprocs} ./exe
*/

# include <iostream>
# include <random>
# include <mpi.h>

void generate_vector(double* a, int N){
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_real_distribution<double> dist(0,100);
    for (int i=0; i<N; i++){
        a[i] = dist(gen);
    }
}
void print_v (double* a, int N){
    for (int i=0; i<N; i++){
        std::cout << a[i] << " ";
    }
}

int main (int argc, char** argv){
    int nprocs, rank;
    int N;
    double* a = nullptr;
    double* b = nullptr;

    MPI_Init(&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout << "Dimension of Vectors: ";
        std::cin>> N;
        if (N<=0){
            std::cerr<<"Dimension Error: Dimension must be positive."<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        a = new double [N];
        b= new double [N];

        generate_vector(a,N);
        generate_vector(b,N);

        std::cout<<"a:";
        print_v(a,N);
        std::cout<<std::endl;
        std::cout<<"b:";
        print_v(b,N);
        std::cout<<std::endl;
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_N = (N/nprocs) + (rank < N%nprocs ? 1 : 0);

    double* local_a = new double[local_N];
    double* local_b = new double[local_N];

    int* cnt = new int[nprocs];
    int* dis = new int[nprocs];

    for (int i=0; i<nprocs; i++){
        cnt[i] = (N/nprocs) + (i < N%nprocs ? 1 : 0);

        if (i == 0){
            dis[i] = 0;
        }else{
            dis[i] = dis[i-1] + cnt[i-1];
        }
    }

    MPI_Scatterv (a, cnt, dis, MPI_DOUBLE,
                  local_a, local_N, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
    
    MPI_Scatterv (b, cnt, dis, MPI_DOUBLE,
                  local_b, local_N, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);    

    double* local_c = new double [local_N];
    for (int i=0; i<local_N; i++){
        local_c[i] = local_a[i] + local_b[i];
    }

    double* global_c = new double[N];

    MPI_Allgatherv (local_c, local_N, MPI_DOUBLE,
                    global_c, cnt, dis, MPI_DOUBLE,
                    MPI_COMM_WORLD);
    
    double time_1 = MPI_Wtime();
    
    if (rank == 0){
        std::cout<<"a+b: "<<std::endl;
        print_v(global_c, N);
        std::cout<<std::endl;

        std::cout<<"Time elapsed:  "<<time_1-time_0<<" seconds"<<std::endl;
        delete [] a;
        delete [] b;
    }

    delete[]  local_a;
    delete[]  local_b;
    delete[]  local_c;
    delete[]  global_c;
    delete[]  cnt;
    delete[]  dis;

    MPI_Finalize();
    return 0;
}
