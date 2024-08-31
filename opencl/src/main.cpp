#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>

#include "host_code.hpp"
#include "utils.hpp"
#include "Timer.hpp"
#include "sequential.hpp"
#include "device_code.hpp"

cl::Context context;
cl::CommandQueue q;
cl::Program program;

Timer timer;
Sequential sequential;


void prepare(int rank){
    std::cout << std::endl;
    timer.start();
    // Rank % 2 is used to select the FPGA device between the two on each node
    prepare(context, q, program, rank % 2);
    timer.stop();
    if(rank == 0){
        std::cout<< "Prepare OpenCL time:\t";
        timer.print_last_formatted();
        std::cout<< "\n";
    }
}

void full_test(std::string n_str, int rank, int size) {

    double *A, *b, *x_fpga;

    size_t n, m;
    int max_iters = 1000;
    double rel_error = 1e-6;
    std::string matrix_path = "/project/home/p200301/io/matrix" + n_str + ".bin";
    std::string rhs_path = "/project/home/p200301/io/rhs"+ n_str + ".bin";
    utils::read_matrix_from_file(matrix_path.c_str(), A, n, m);
    utils::read_vector_from_file(rhs_path.c_str(), b, n);

    x_fpga = new (std::align_val_t{64}) double[n];

    auto start = std::chrono::high_resolution_clock::now();
    conjugate_gradient(context, q, program, A, b, x_fpga, n, max_iters, rel_error, rank, size); 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if(rank==0)
        std::cout << elapsed.count() << std::endl;

    delete[] A;
    delete[] b;
    delete[] x_fpga;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    prepare(rank);
    MPI_Barrier(MPI_COMM_WORLD);


    std::vector<std::string> n (argv  + 1, argv + argc - 1);

    int K = std::stoi(argv[argc-1]);

    for(auto &n_str : n ){
        if(rank == 0)
            std::cout<< "N: " << n_str << std::endl;
        for(int i=0; i<K; i++){
            full_test(n_str, rank, size);
        }
    }

    MPI_Finalize();
	return 0;
}
