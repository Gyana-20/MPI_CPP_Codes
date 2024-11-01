/*
Program:
- This program demonstrates the implimentation of 
  Parallel Matrix vector Multiplication using 
  block row decomposition.

Author: GYANA RANJAN NAYAK

Usage:
mpic++ -o {exe} {filename.cpp}  
mpirun -np {nprocs} ./exe
*/

# include <iostream>
# include<random>
# include <mpi.h>

void generate_matrix (double* A, int A_r, int A_c){
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_real_distribution<double>dist1(0,100);
    for (int i=0; i<A_r; i++){
        for (int j=0; j<A_c; j++){
            A[i*A_c+j] = dist1(gen);
        }
    }
}

void generate_vector(double* x, int x_r){
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_real_distribution<double>dist2(0,100);
    for (int i=0; i<x_r; i++){
        x[i] = dist2(gen);
    }
}

void print_m (double* A, int A_r, int A_c){
    for (int i=0; i<A_r; i++){
        for (int j=0; j< A_c; j++){
            std::cout<<A[i*A_c + j]<<" ";
        }
        std::cout<<std::endl;
    }
}
void print_v (double* x, int x_r){
    for (int i=0; i<x_r; i++){
        std::cout<<x[i]<<" ";
    }
}

int main (int argc, char** argv){
    int nprocs, rank;
    int Ar, Ac, xr;
    double* A;
    double* x;

    MPI_Init(&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"Enter dimension of the matrix (rows followed by cols): "<<std::endl;
        std::cin>> Ar>> Ac;

        std::cout<<"Enter the dimension of the vector: ";
        std::cin>>xr;

        if (Ar <=0 || Ac<=0 || xr <=0){
            std::cerr<<"Dimension Error :  All dimensions must be positive integers."<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (Ac != xr){
            std::cerr<<"Dimension Error :  number of cols  in matrix must be equal to the dimension of the vector."<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A = new  double[Ar*Ac];
        x = new double[xr];

        generate_matrix(A,  Ar, Ac);
        generate_vector(x, xr);

        // std::cout<<"A: "<<std::endl;
        // print_m(A,Ar,Ac);
        // std::cout<<"\n";

        // std::cout<<"x: "<<std::endl;
        // print_v(x,xr);
        // std::cout<<"\n";
    }

    MPI_Bcast(&Ar, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Ac, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_Ar = Ar/nprocs + (rank < Ar%nprocs ? 1:0);
    double* local_A = new double[local_Ar * Ac];
    
    int local_xr = xr/nprocs + (rank < xr%nprocs ? 1:0);
    double* local_x = new double[local_xr];

    int* cnt_A = new int [nprocs];
    int* dis_A =  new int [nprocs];
    int* cnt_x = new int [nprocs];
    int* dis_x = new int [nprocs];

    for (int i=0; i< nprocs; i++){
        cnt_A[i] = (Ar/nprocs + (i < Ar%nprocs ? 1:0))*Ac;
        cnt_x[i] =  (xr/nprocs + (i < xr%nprocs ? 1:0));
        if (i==0){
            dis_A[i]=0;
            dis_x[i]=0;
        }else{
            dis_A[i] = dis_A[i-1] +  cnt_A[i-1];
            dis_x[i]  = dis_x[i-1] + cnt_x[i-1];
        }
    }

    MPI_Scatterv(A, cnt_A, dis_A, MPI_DOUBLE, 
                 local_A, local_Ar*Ac, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(x, cnt_x, dis_x, MPI_DOUBLE, 
                 local_x, local_xr, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double* global_x = new  double[xr];

    MPI_Allgatherv(local_x, local_xr,MPI_DOUBLE,
                    global_x, cnt_x, dis_x, MPI_DOUBLE,
                    MPI_COMM_WORLD);
    
    double* local_Ax = new double[local_xr];
    for (int i=0; i<local_xr; i++){
        local_Ax[i] = 0;
        for (int j=0; j<Ac;  j++){
            local_Ax[i] += local_A[i*Ac + j]*global_x[j];
        }
    }
    double* global_Ax = new double[xr];
    MPI_Allgatherv(local_Ax, local_xr,MPI_DOUBLE,
                    global_Ax, cnt_x, dis_x, MPI_DOUBLE,
                    MPI_COMM_WORLD);
    
    double time_1 = MPI_Wtime();

    if (rank == 0){
        // std::cout<<"Ax: "<<std::endl;
        // print_v(global_Ax,  xr);
        // std::cout<<std::endl;
        std::cout<<"Program finished with no errors."<<std::endl;
        std::cout<<"Elapsed Time: "<<time_1 - time_0 << " sec"<<std::endl;
        delete[] A;
        delete[] x;
    }

    delete[] cnt_A;
    delete[] dis_A;
    delete[] cnt_x;
    delete[] dis_x;
    delete[] local_A;
    delete[] local_x;
    delete[] global_x;
    delete[] local_Ax;
    delete[] global_Ax;

    MPI_Finalize();
    return 0;
}