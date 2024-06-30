#include <iostream>
#include <vector>
#include <gtest/gtest.h>


#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

//#include "test_template.hpp"
#include "cgcore.hpp"

using namespace cgcore;

bool AreArraysEqual(double* arr1, double* arr2, size_t size, double tol, double &norm_diff) {

    norm_diff = 0.0;

    for (size_t i = 0; i < size; ++i) {
        norm_diff += std::abs(arr1[i] - arr2[i]);
    }

    return norm_diff <= tol;
}


void full_test(std::string n_str){

    const std::string _m_path = "/project/home/p200301/tests/matrix" + n_str + ".bin"; 	
    const std::string _rhs_path  = "/project/home/p200301/tests/rhs" + n_str + ".bin"; 	
    const char *m_path = _m_path.c_str();
    const char *rhs_path = _rhs_path.c_str();


    double *matrix;
    double *vector;
    double *x_seq, *x_fpga;

    size_t n, m ; 
    int max_iter = 10000;
    double res = 1.e-4;

    CGSolver<Sequential> seq_solver;
    CGSolver<FPGA_CG> fpga_solver;

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);

    std::cout << n << std::endl;
    x_seq = new (std::align_val_t{64}) double[n];
    x_fpga = new (std::align_val_t{64}) double[n];
        
    seq_solver.solve(matrix, vector, x_seq, n, max_iter, res);
    fpga_solver.solve(matrix, vector, x_fpga, n, max_iter, res);

    double norm_diff;
    //EXPECT_TRUE(AreArraysEqual(x_seq, x_fpga, n, res, norm_diff)) << "Fail";
    AreArraysEqual(x_seq, x_fpga, n, res, norm_diff);
    std::cout << "Norm diff: " << norm_diff << std::endl;

    delete [] matrix;
    delete [] vector;
    delete [] x_seq;
    delete [] x_fpga;

    seq_solver.get_timer().print_last_formatted() ;
    fpga_solver.get_timer().print_last_formatted() ;

}

int main(int argc, char* argv[]) {

	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << "n1 n2 .. ni" << std::endl;
		return 1;
	}
	std::vector<std::string> n_str(argv + 1, argv + argc);

	for(const auto& n: n_str){
		full_test(n);
	}

	return 0;
}
