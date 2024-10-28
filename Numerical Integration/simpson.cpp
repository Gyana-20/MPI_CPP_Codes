/*
Program:
Evaluation of PI using Simpson's rule,
by the integration of 4/(1+x*x).
(Enter a = 0, b = 1)

MPI_Reduce, MPI_Bcast have been used.

Author:
Gyana Ranjan Nayak

Usage:
mpic++ -o simpson simpson.cpp
mpirun -np 4 ./simpson
*/

# include <iostream>
# include <mpi.h>
# include <cmath>
# include <iomanip>

double f (double x){
    return  4/(1+pow(x,2));
}

double simpson (double a, double b, int n, double h){
    double even_sum = 0;
    double odd_sum = 0;

    for (int i = 0; i < n; i++){
        if (i%2 == 0){
            even_sum += 2*f(a + i*h);
        }
        else {
            odd_sum += 4*f(a + i*h);
        }
    }
    return (even_sum + odd_sum);
}

int main (int argc, char** argv){
    
    double a, b, h;
    int n;
    int rank, nprocs;
    double local_a, local_b, local_n;
    double local_sum, global_sum;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init (&argc, &argv);

    MPI_Comm_size (comm, &nprocs);
    MPI_Comm_rank (comm, &rank);

    double time0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"a: ";
        std::cin>>a;
        std::cout<<"\n";

        std::cout<<"b: ";
        std::cin>>b;
        std::cout<<"\n";

        std::cout<<"n: ";
        std::cin>>n;
        std::cout<<"\n";
    }
    MPI_Bcast (&a, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast (&b, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast (&n, 1, MPI_INT, 0, comm);

    h = (b-a)/n ;

    local_n = n/nprocs;
    local_a = a + rank*local_n*h;
    local_b = local_a + local_n*h;

    if (rank == 0) local_a = a+h;
    if (rank == nprocs-1) local_b = b-h;

    local_sum = simpson (local_a, local_b, local_n, h);

    MPI_Reduce (&local_sum, &global_sum, 1, MPI_DOUBLE, 
                MPI_SUM, 0, comm);
    
    double time1 = MPI_Wtime();

    if (rank == 0){
        double final_sum = (f(a) + f(b)+global_sum)*(h/3);

        double time2 = MPI_Wtime();

        std::cout << std::fixed<<std::setprecision(10);
        std::cout<<"Integral value: "<<final_sum<<std::endl;

        std::cout<<"Total time elapsed (in sec) : "<<time2 - time0<<std::endl;

        std::cout<<"Parallel computation time (in sec) : "<<time1 - time0 <<std::endl;
    }

    MPI_Finalize();
    return 0;
}