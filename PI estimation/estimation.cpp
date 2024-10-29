/*
mpic++ -o run1 estimation.cpp
mpirun -np 4 ./run1
*/

# include <iostream>
# include <iomanip>
# include <random>
# include <mpi.h>


int main (int agrc, char** argv){
    int rank, nprocs;
    int N, local_count = 0, global_count = 0;

    MPI_Init (&agrc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"Enter Number of sample points: ";
        std::cin>>N;
        std::cout<<std::endl;

        if (N<=0){
            std::cerr<<"Err: Number of sample points must be positive."<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_el = N/nprocs;
    int rem_el = N%nprocs;
    int local_N = base_el + (rank < rem_el ? 1 : 0);

    std:: random_device device;
    std:: mt19937 gen(device());
    std:: uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i=0; i<local_N; i++){
        double x = dist(gen);
        double y = dist(gen);
        if (x*x + y*y <=1){
            local_count ++;
        }
    }

    MPI_Reduce(&local_count, &global_count, 1,
                MPI_INT, MPI_SUM, 0,
                MPI_COMM_WORLD);
    
    double time_1 = MPI_Wtime();

    if (rank == 0){
        std::cout<<std::fixed<<std::setprecision(15);
        double pi =  double(4.0*global_count)/double(N);
        std::cout<<"Approximation of PI : "<< pi << std::endl;
        std::cout<<"Parallel computation time: "<<time_1- time_0 <<" sec"<<std::endl;
    }

    MPI_Finalize();
    return 0;
}