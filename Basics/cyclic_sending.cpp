/*
Program:
- This program demonstrates basic message passing
  between the processes.

- process i sends the message to the process (i+1)%nprocs
  where nprocs is the total number of processes.

- Rank i (!=0) processor receives the message from (i-1)th rank
  processor. If i = 0, then it receives from the processor
  with  rank nprocs-1.

Author: GYANA RANJAN NAYAK

Usage:
mpic++ -o {exe}  {filename.cpp}
mpirun -np {nprocs} ./{exe}
*/

# include <iostream>
# include <string>
# include <mpi.h>

int main(int argc, char** argv){
    int nprocs, rank;
    MPI_Status status;

    MPI_Init (&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    std::string greetings = std::string("Hello from ") + std::to_string (rank);

    int length =  greetings.length();

    MPI_Send (greetings.c_str(), length+1, MPI_CHAR,
              (rank+1)%nprocs, 101, MPI_COMM_WORLD);

    char* buffer = new char[length+1];
    
    if (rank == 0){
        MPI_Recv (buffer, length+1, MPI_CHAR, 
                  nprocs-1, 101, MPI_COMM_WORLD, &status);
    }
    else{
        MPI_Recv (buffer, length+1, MPI_CHAR,
                  rank-1, 101, MPI_COMM_WORLD, &status);
    } 

    std::cout<<"Rank "<<rank<<" || message received: "<<buffer<<std::endl;

    delete[] buffer;
    
    MPI_Finalize();
    return 0;
}