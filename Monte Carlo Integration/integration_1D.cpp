/*
Program:
- Monte carlo Integration of e(-x*x/2) from 0 to 1

Author:
Gyana Ranjan Nayak

Usage:
mpic++ -o {exe} {filename.cpp}
mpirun -np {nprocs} ./exe

*/

# include <iostream>
# include <mpi.h>
# include <random>
# include <cmath>
# include <iomanip>

double h (double x){
    return exp(-x*x/2);
}

int main (int argc, char** argv){
    int rank, nprocs;
    int N; // number of points
    double* x = nullptr;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"Enter the number of points: ";
        std::cin>>N;
        if (N<=0){
            std::cerr<<"Err: number of points must be positive"<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::random_device device;
        std::mt19937  gen(device());
        std::uniform_real_distribution<> dist(0,1);

        x = new double [N];

        for (int i = 0; i<N; i++){
            double random_value = dist(gen);
            x[i] = random_value;
        }

    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_el = N/nprocs;
    int rem_el = N%nprocs;
    int local_N = base_el + (rank < rem_el ? 1 : 0);

    double* local_x = new double [local_N];

    int* send_cnt = new int [nprocs];
    int* send_dis = new int [nprocs];

    for (int i = 0; i<nprocs; i++){
        send_cnt[i] = base_el + (i < rem_el ? 1 : 0);

        if (i == 0){
            send_dis[i] = 0;
        }else{
            send_dis[i] = send_dis[i-1] + send_cnt[i-1];
        }
    }

    MPI_Scatterv (x, send_cnt, send_dis, 
                  MPI_DOUBLE, local_x, local_N, 
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0;
    for (int i = 0; i< local_N; i++){
        local_sum += h(local_x[i]);
    }
    
    double global_sum = 0;

    MPI_Reduce(&local_sum, &global_sum, 1, 
                MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
    
    double time_1 = MPI_Wtime();

    if (rank == 0){

        std::cout<<std::fixed<<std::setprecision(15);

        std::cout<<"Integral of e^(-x*x/2) from 0 to 1: "<<std::endl;
        std::cout<< global_sum/double(N) <<std::endl;

        std::cout<<"Parallel Computation time: "<<time_1 - time_0 <<" sec"<<std::endl;
        delete[] x;
    }

    delete[] local_x;
    delete[] send_cnt;
    delete[] send_dis;

    MPI_Finalize();
    return 0;

}