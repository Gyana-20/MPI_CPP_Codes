/*
Program:
- This program demonstrates the implimentation of 
  Parallel matrix addition using Block row decomposition.

Author: GYANA RANJAN NAYAK

Usage:
mpic++ -o {exe} {filename.cpp}  
mpirun -np {nprocs} ./exe
*/



# include <iostream>
# include <random>
# include <mpi.h>

void generate_matrix (double* A, int rows, int cols){
    std:: random_device device;
    std:: mt19937 gen (device());
    std:: uniform_real_distribution<double> dist(0,100);

    for (int i=0; i<rows; i++){
        for (int j=0; j<rows; j++){
            A[i*cols+j] = dist(gen);
        }
    }
}

void print_m (double* A, int rows, int cols){
    for (int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            std::cout<<A[i*cols + j]<<" ";
        }
        std::cout<<std::endl;
    }
}

int main(int argc, char** argv){
    int nprocs, rank;
    int rows, cols;
    double* A;
    double* B;

    MPI_Init(&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    double time_0 = MPI_Wtime();

    if (rank == 0){
        std::cout<<"Number of rows followed by cols: "<<std::endl;
        std::cin>> rows>>cols;

        if (rows <=0 || cols<=0){
            std::cerr<<"Error: Number of rows and cols must be positive."<<std::endl;
            MPI_Abort (MPI_COMM_WORLD, 1);
        }

        A = new double [rows* cols];
        B = new double [rows* cols];

        generate_matrix(A,rows,cols);
        generate_matrix(B,rows,cols);

    }

    MPI_Bcast (&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = (rows/nprocs) + (rank < rows%nprocs ? 1 : 0);

    double* local_A = new double[local_rows* cols];
    double* local_B = new double[local_rows* cols];

    int* cnt = new int [nprocs];
    int* dis = new int [nprocs];

    for (int i=0; i<nprocs; i++){
        cnt[i] = ((rows/nprocs) + (i < rows%nprocs ? 1 : 0))*cols;
        if (i == 0){
            dis[i] = 0;
        }else{
            dis[i] = dis[i-1] + cnt[i-1];
        }
    }

    double* local_C = new double [local_rows*cols];

    MPI_Scatterv (A, cnt, dis, MPI_DOUBLE,
                  local_A, local_rows*cols, 
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv (B, cnt, dis, MPI_DOUBLE,
                  local_B, local_rows*cols, 
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int i=0; i<local_rows; i++){
        for(int j=0; j<cols; j++){
            local_C[i*cols + j] = local_A[i*cols + j] + local_B[i*cols + j];
        }
    }

    double* global_C = new double[rows*cols];

    MPI_Allgatherv (local_C, local_rows*cols, MPI_DOUBLE,
                    global_C, cnt, dis, MPI_DOUBLE,
                    MPI_COMM_WORLD);
    
    double time_1 = MPI_Wtime();


    if (rank == 0){
        std::cout<<"Program executed successfully!"<<std::endl;
        std::cout<<"Elapsed Time "<<time_1 - time_0 <<" seconds"<<std::endl;
        delete[] A;
        delete[] B;
    }

    delete[] local_A;
    delete[] local_B;
    delete[] local_C;
    delete[] cnt;
    delete[] dis;
    
    MPI_Finalize();
    return 0;
}
