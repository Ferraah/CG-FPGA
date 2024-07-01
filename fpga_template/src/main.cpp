#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "host_code.hpp"
#include "utils.hpp"
#include "Timer.hpp"
#include "sequential.hpp"

#include "device_code.hpp"

void full_test(std::string n_str){

    const std::string _m_path = "/project/home/p200301/tests/matrix" + n_str + ".bin"; 	
    const std::string _rhs_path  = "/project/home/p200301/tests/rhs" + n_str + ".bin"; 	
    const char *m_path = _m_path.c_str();
    const char *rhs_path = _rhs_path.c_str();


    double *matrix;
    double *vector;
    double *x_fpga, *x_seq;

    size_t n=10, m=10 ; 
    int max_iter = 10000;
    double res = 1.e-4;

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);
    //utils::create_matrix(matrix, n, m, 1);
    //utils::create_vector(vector, n, 1);

    std::cout << n << std::endl;
    x_fpga = new (std::align_val_t{64}) double[n];
    x_seq = new (std::align_val_t{64}) double[n];

    sycl::queue q;
    Timer timer;
    Sequential sequential;

    //test(A, b);
    //return;

    std::cout << std::endl;
    timer.start();
    prepare(q);
    timer.stop();
    timer.print_last_formatted();

    std::cout << std::endl;
    timer.start();
    conjugate_gradient(q, matrix, vector, x_fpga, n, max_iter, res);
    timer.stop();
    timer.print_last_formatted();

    std::cout << std::endl;
    timer.start();
    sequential.conjugate_gradient(matrix, vector, x_seq, n, max_iter, res);
    timer.stop();
    timer.print_last_formatted();



    //double norm_diff;
    //EXPECT_TRUE(AreArraysEqual(x_seq, x_fpga, n, res, norm_diff)) << "Fail";
    //AreArraysEqual(x_seq, x_fpga, n, res, norm_diff);
    //std::cout << "Norm diff: " << norm_diff << std::endl;

    delete [] matrix;
    delete [] vector;
    delete [] x_seq;
    delete [] x_fpga;

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