#include <iostream>
#include <vector>

#include "host_code.hpp"
#include "utils.hpp"
#include "Timer.hpp"
#include "sequential.hpp"
#include "device_code.hpp"

cl::Context context;
cl::Device dev;
cl::CommandQueue q;
cl::Program program;
cl::Program fblas_program;

Timer timer;
Sequential sequential;

void full_test(std::string n_str){

    const std::string _m_path = "/project/home/p200301/tests/matrix" + n_str + ".bin"; 	
    const std::string _rhs_path  = "/project/home/p200301/tests/rhs" + n_str + ".bin"; 	
    const char *m_path = _m_path.c_str();
    const char *rhs_path = _rhs_path.c_str();


    double *matrix;
    double *vector;
    double *x_fpga, *x_seq;

    size_t n=10, m=10 ; 
    int max_iter = 1000;
    double res = 1.e-4;

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);
    //utils::create_matrix(matrix, n, m, 1);
    //utils::create_vector(vector, n, 1);

    std::cout << n << std::endl;
    x_fpga = new (std::align_val_t{64}) double[n];
    x_seq = new (std::align_val_t{64}) double[n];
    
    dev = cl::Device::getDefault();

    std::cout << std::endl;
    timer.start();
    conjugate_gradient(dev, context, q, program, fblas_program, matrix, vector, x_fpga, n, max_iter, res);
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
void prepare(){
    std::cout << std::endl;
    timer.start();
    prepare(context, q, program, fblas_program);
    timer.stop();
    timer.print_last_formatted();
}

void find_opencl(){
    // Get all platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return;
    }

    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // Get all devices for the current platform
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (devices.empty()) {
            std::cout << "No devices found for platform " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
            continue;
        }

        for (const auto& device : devices) {
            std::cout << "  Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "    Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "    Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
            std::cout << "    Driver Version: " << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
            std::cout << "    Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "    Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "    Global Memory Size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << " bytes" << std::endl;
            std::cout << std::endl;
        }
    }
}

void gemv_test(int n) {

    n = 4;
    //double *A, *x;
    double A[16] = {-1,2,-3,4,0,3,1,2,4,5,1,6,1,2,3,4};
    double x[4] = {1, -2, 3, -4};
    double b_ref[4] = {-30, -11, -27, -10};

    auto *b= new (std::align_val_t{64}) double[n];


    //utils::create_matrix(A, n, n, 1);
    //utils::create_vector(x, n, 1);

    for(int c=0; c<n; c++){
        //A[(n-1)*n + c] = 0;
    }

    cl_int err;
    //cl::Buffer d_A(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , n*n*sizeof(double), (void*)A, &err);
    //cl::Buffer d_x(context, CL_CHANNEL_2_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)x, &err);
    //cl::Buffer d_b(context, CL_CHANNEL_3_INTELFPGA | CL_MEM_WRITE_ONLY, n*sizeof(double));
    cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , n*n*sizeof(double), (void*)A, &err);
    cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)x, &err);
    cl::Buffer d_b(context, CL_MEM_WRITE_ONLY, n*sizeof(double));
    CHECK_ERR(err, "err gemv test");

    cl::Event event;

    PipesDeviceHandler device(dev, context, q, program, fblas_program);
    device.matrix_vector_mul(d_A, d_x, d_b, n, event);
    q.enqueueReadBuffer(d_b, CL_TRUE, 0, n*sizeof(double), b);
    utils::print_matrix(b, 1, n);
    for (int i = 0; i < n; i++)
        if(b[i] != b_ref[i]){
            std::cerr << "gemv not passed\n";
            return;
        }

    /*
    std::cout << std::endl;
    utils::print_matrix(A, n, n);
    std::cout << std::endl;
    utils::print_matrix(b, 1, n);
    std::cout << std::endl;
    //ASSERT_EQ(h_res[0], (double)n);
*/
    std::clog << "gemv passed\n";

    //delete[] A;
    delete[] b;
    //delete[] x;

}

void gemv_full_test(int n) {

    double *A, *x;

    auto *b= new (std::align_val_t{64}) double[n];


    utils::create_matrix(A, n, n, 1);
    utils::create_vector(x, n, 1);

    cl_int err;
    //cl::Buffer d_A(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , n*n*sizeof(double), (void*)A, &err);
    //cl::Buffer d_x(context, CL_CHANNEL_2_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)x, &err);
    //cl::Buffer d_b(context, CL_CHANNEL_3_INTELFPGA | CL_MEM_WRITE_ONLY, n*sizeof(double));
    cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , n*n*sizeof(double), (void*)A, &err);
    cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)x, &err);
    cl::Buffer d_b(context, CL_MEM_WRITE_ONLY, n*sizeof(double));
    CHECK_ERR(err, "err gemv test");

    cl::Event event;

    PipesDeviceHandler device(dev, context, q, program, fblas_program);
    device.matrix_vector_mul(d_A, d_x, d_b, n, event);
    q.enqueueReadBuffer(d_b, CL_TRUE, 0, n*sizeof(double), b);
    //utils::print_matrix(b, 1, n);
    for (int i = 0; i < n; i++)
        if(b[i] != n){
            std::cerr << "full gemv not passed\n";
            return;
        }

    /*
    std::cout << std::endl;
    utils::print_matrix(A, n, n);
    std::cout << std::endl;
    utils::print_matrix(b, 1, n);
    std::cout << std::endl;
    //ASSERT_EQ(h_res[0], (double)n);
*/
    std::clog << "full gemv passed\n";

    //delete[] A;
    delete[] b;
    //delete[] x;

}

void vecsum_test(int n) {

    auto *a= new (std::align_val_t{64}) double[n];

    auto *c= new (std::align_val_t{64}) double[n];

    for (int i = 0; i < n; i++)
        a[i] = 1.0;

    cl_int err;
    cl::Buffer d_a(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)a, &err);
    cl::Buffer d_b(context, CL_CHANNEL_2_INTELFPGA | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)a, &err);
    CHECK_ERR(err, "err vecsum test");

    cl::Event event;

    PipesDeviceHandler device(dev, context, q, program, fblas_program);
    device.vec_sum(1.0, d_a, 1.0, d_b, n, event);
    q.enqueueReadBuffer(d_b, CL_TRUE, 0, n*sizeof(double), c);
    for (int i = 0; i < n; i++)
        if(c[i] != 2.0){
            std::cerr << "vec sum not passed\n";
            return;
        }

    std::clog << "vec sum passed\n";

    delete[] a;
    delete[] c;

}



void dot_test(int n) {

    auto *a= new (std::align_val_t{64}) double[n];

    auto *c= new (std::align_val_t{64}) double[1];

    for (int i = 0; i < n; i++)
        a[i] = 1.0;

    cl_int err;
    cl::Buffer d_a(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, n*sizeof(double), (void*)a, &err);
    cl::Buffer d_b(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, n*sizeof(double), (void*)a, &err);
    cl::Buffer d_c(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_WRITE_ONLY, sizeof(double));
    CHECK_ERR(err, "err dot test");


    std::vector<cl::Event> events;

    PipesDeviceHandler device(dev, context, q, program, fblas_program);
    device.dot(d_a, d_b, d_c, n, events);
    q.enqueueReadBuffer(d_c, CL_TRUE, 0, sizeof(double), c);
    if(c[0] != n)
        std::cerr << "Dot test not passed\n";
    else
        std::clog << "Dot test passed\n";

    //ASSERT_EQ(h_res[0], (double)n);


    delete[] a;
    delete[] c;

}

int main(int argc, char* argv[]) {

    
	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << "n1 n2 .. ni" << std::endl;
		return 1;
	}
	std::vector<std::string> n_str(argv + 1, argv + argc);

    prepare();
     int n = 512;

    //gemv_full_test(n);
    dot_test(n);
    vecsum_test(n);
  //  gemv_test(n);
  //  
  //  dot_test(2*n);
  //  vecsum_test(2*n);
  //  gemv_test(2*n);
    //find_opencl();
	for(const auto& n: n_str){
//		full_test(n);
	}
	return 0;
}
