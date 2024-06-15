#include <iostream>
#include <vector>
#include <gtest/gtest.h>


#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

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
int usm_selector(const sycl::device& dev) {
  if (dev.has(sycl::aspect::usm_device_allocations)) {
    return 1;
  }
  return -1;
}

sycl::queue create_queue(){
    #if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
    std::cout << "Using sim settings";
    #elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
    std::cout << "Using HW settings";
    #else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    std::cout << "Using emu settings";
    #endif

    // create the device queue
    sycl::queue q(selector, sycl::property::queue::in_order{} );
    // make sure the device supports USM host allocations
    auto device = q.get_device();

    std::clog << "Running on device: "
                << device.get_info<sycl::info::device::name>().c_str()
                << std::endl;
            
    std::clog << "Supports USM: " << (bool)usm_selector(device)<<std::endl;
    return q;
 

}

TEST(CG_TEST, vec_sum){
    sycl::queue q = create_queue(); 
    constexpr int n = 1024;

    auto d_vec = sycl::malloc_shared<double>(n*sizeof(double), q); 
    double *expected;

    for(int i=0; i<n; i++) 
        d_vec[i] = 1.0; 
    
    utils::create_vector(expected, n, 2.0);

    FPGA_CG strategy;
    strategy.vec_sum(q, 1.0, d_vec, 1.0, d_vec, n);
    q.wait();

    ASSERT_TRUE(AreArraysEqual(d_vec, expected, n, 0));

    sycl::free(d_vec, q);
    delete [] expected;

}

TEST(CG_TEST, dot){
    sycl::queue q = create_queue(); 
    constexpr int n = 1024;

    auto d_vec = sycl::malloc_shared<double>(n*sizeof(double), q); 
    auto d_res = sycl::malloc_shared<double>(1*sizeof(double), q); 
    
    for(int i=0; i<n; i++) 
        d_vec[i] = 1;
    
    FPGA_CG strategy;
    strategy.dot(q, d_vec, d_vec, d_res, n);
    q.wait();

    ASSERT_EQ(d_res[0], (double)n);
    sycl::free(d_vec, q);
    sycl::free(d_res, q);

}

TEST(CG_TESTS, full){
    const char *m_path =  "/project/home/p200301/tests/matrix1000.bin";
    const char *rhs_path =  "/project/home/p200301/tests/rhs1000.bin";

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