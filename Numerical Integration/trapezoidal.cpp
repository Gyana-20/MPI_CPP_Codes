/*

Program:
Evaluation of PI using Trapezoidal rule,
by the integration of 4/(1+x*x).


MPI_Reduce, MPI_Bcast have been used.

Author:
Gyana Ranjan Nayak

Usage:
mpic++ -o trapezoidal trapezoidal.cpp
mpirun -np 4 ./trapezoidal
*/

# include <mpi.h>
# include <iostream>
# include <cmath>
# include <iomanip>

double f(double x){
    return 4/(1+pow(x,2));
}

double trapezoidal (double a, double b, int n, double h){
    double s = 0;
    for (int i=0; i<=n; i++){
        s += f(a+i*h);
    }
    return s;
}

int main(int argc, char** argv){
    int nprocs, rank;
    int n;
    double a = 0, b = 1, h;
    
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);

    MPI_Comm_size (comm, &nprocs);
    MPI_Comm_rank (comm, &rank);

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"n: ";
        std::cin>>n;
        std::cout<<"\n";

        h= (b-a)/n;
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, comm);
    
    int local_n = n/nprocs;
    double local_a = a+rank*h*local_n;
    double local_b = local_a + h*local_n;
    if (rank == nprocs-1){
        local_b =b-h;
    }

    if (rank == 0){
        local_a = a+h;
    }

    double local_sum = trapezoidal(local_a, local_b, local_n, h);

    double global_sum;

    MPI_Reduce (&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    0, comm);

    double time_1 = MPI_Wtime();

    if (rank == 0){
        std::cout<<std::fixed<<std::setprecision(10)<<std::endl;

        double final_sum = ((f(a)+f(b))/2 + global_sum)*h;

        double time_2 = MPI_Wtime();
        
        std::cout<<"integral of 4/(1+x^2) from 0 to PI: "
                  << final_sum<<std::endl;
        
        std::cout<<"Total time elapsed (in sec): "<<time_2-time_0<<std::endl;
        std::cout<<"Time elapsed in Parallel section: "<<time_1 -time_0 <<std::endl;

    }

    MPI_Finalize();
    return 0;
}