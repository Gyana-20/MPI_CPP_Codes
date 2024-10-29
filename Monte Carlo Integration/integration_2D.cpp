/*
Program:
- Monte carlo Integration of e(-x*x-y*y) 
- x ranging from 0 to 1
- y ranging from 0 to 1

Author:
Gyana Ranjan Nayak

Usage:
mpic++ -o {exe} {filename.cpp}
mpirun -np {nprocs} ./exe
*/


# include <iostream>
# include <iomanip>
# include <cmath>
# include <random>
# include <mpi.h>

double f (double x, double y){
    return exp(-x*x -y*y);
}

int main (int argc, char** argv){
    int nprocs, rank;
    int N; // assumning number of points in both the x and y directions are same
    double a = 0.0, b = 1.0;
    double V = pow((b-a),2);

    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"Enter number of sample points: "<<std::endl;
        std::cin>>N;
        if (N<=0){
            std::cerr<<"Err: number of sample points must be positive"<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
    }
    
    MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_N = (N/nprocs) + (rank < N%nprocs ? 1:0);
    double local_sum = 0;
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_real_distribution<double> dist(a, b);

    for (int i = 0; i<local_N; i++){
        double x = dist(gen);
        double y = dist(gen);
        local_sum += f(x,y);
    }

    double global_sum = 0;
    MPI_Reduce (&local_sum, &global_sum, 1, 
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double time_1 = MPI_Wtime();
    
    if (rank == 0){
        std::cout<<std::fixed<<std::setprecision(15);

        std::cout<<"Integral of e^(-x*x-y*y) x = 0 to 1, y = 0 to 1 :"<<std::endl;
        std::cout<< V*(global_sum/double(N))<< std::endl;

        std::cout<<"Time elapsed: "<<time_1 - time_0 <<" sec"<<std::endl;
    }

    MPI_Finalize();
    return 0;
}