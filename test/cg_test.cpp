#include <iostream>
#include <vector>
#include <gtest/gtest.h>

//#include "test_template.hpp"
#include "cgcore.hpp"

using namespace cgcore;

// Function to compare two dynamic arrays
bool AreArraysEqual(double* arr1, double* arr2, size_t size, double tol) {

    for (int i = 0; i < size; ++i) {
        if (std::abs(arr1[i] - arr2[i]) > tol) {
            return false;
        }
    }
    return true;
}

TEST(GEMV, gemv_1){
    const char *m_path =  "/project/home/p200301/tests/matrix100.bin";
    const char *rhs_path =  "/project/home/p200301/tests/rhs100.bin";

    double *matrix;
    double *vector;
    double *x_seq, *x_fpga;

    size_t n, m ; 
    int max_iter = 10000;
    double res = 1.e-6;

    CGSolver<Sequential> seq_solver;
    CGSolver<FPGA_CG> fpga_solver;

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);

    std::cout << n << std::endl;
    x_seq = new double[n];
    x_fpga = new double[n];
        
    seq_solver.solve(matrix, vector, x_seq, n, max_iter, res);
    fpga_solver.solve(matrix, vector, x_fpga, n, max_iter, res);

    ASSERT_TRUE(AreArraysEqual(x_seq, x_fpga, n, res)) << "Fail";

    delete [] matrix;
    delete [] vector;
    delete [] x_seq;
    delete [] x_fpga;

    seq_solver.get_timer().print_last_formatted() ;
    fpga_solver.get_timer().print_last_formatted() ;

}
