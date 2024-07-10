#include <iostream>
#include <vector>
#include <string>

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

    size_t n=10, m=10; 
    int max_iter = 1000;
    double res = 1.e-9;

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);
    //utils::create_matrix(matrix, n, m, 1);
    //utils::create_vector(vector, n, 1);

    // Initialize solutions
    x_fpga = new (std::align_val_t{64}) double[n];
    x_seq = new (std::align_val_t{64}) double[n];
    
    // FPGA Strategy
    timer.start();
    conjugate_gradient(dev, context, q, program, fblas_program, matrix, vector, x_fpga, n, max_iter, res);
    timer.stop();
    timer.print_last_formatted();

    // Sequential strategy
    timer.start();
    sequential.conjugate_gradient(matrix, vector, x_seq, n, max_iter, res);
    timer.stop();
    timer.print_last_formatted();

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
    double A[16] = {-1,2,-3,4,0,3,1,2,4,5,1,6,1,2,3,4};
    double x[4] = {1, -2, 3, -4};
    double b_ref[4] = {-30, -11, -27, -10};

    auto *b = new (std::align_val_t{64}) double[n];

    cl_int err;
    cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , n*n*sizeof(double), (void*)A, &err);
    cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)x, &err);
    cl::Buffer d_b(context, CL_MEM_WRITE_ONLY, n*sizeof(double));
    CHECK_ERR(err, "err gemv test");


    auto start = std::chrono::high_resolution_clock::now();
    DirectDeviceHandler device(dev, context, q, program, fblas_program);
    device.matrix_vector_mul(d_A, d_x, d_b, n);
    q.enqueueReadBuffer(d_b, CL_TRUE, 0, n*sizeof(double), b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GEMV test time: " << elapsed.count() << " seconds\n";

    utils::print_matrix(b, 1, n);
    for (int i = 0; i < n; i++)
        if(b[i] != b_ref[i]){
            std::cerr << "gemv not passed\n";
            return;
        }

    std::clog << "gemv passed\n";

    delete[] b;
}

void gemv_full_test(int n) {
    double *A, *x;

    auto *b = new (std::align_val_t{64}) double[n];
    auto *y = new (std::align_val_t{64}) double[n];

    utils::create_matrix(A, n, n, 1);
    utils::create_vector(x, n, 1);

    cl_int err;
    cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , n*n*sizeof(double), (void*)A, &err);
    cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)x, &err);
    cl::Buffer d_b(context, CL_MEM_WRITE_ONLY, n*sizeof(double));
    CHECK_ERR(err, "err gemv test");


    auto start = std::chrono::high_resolution_clock::now();
    DirectDeviceHandler device(dev, context, q, program, fblas_program);
    device.matrix_vector_mul(d_A, d_x, d_b, n);
    //q.enqueueReadBuffer(d_b, CL_TRUE, 0, n*sizeof(double), b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "Full GEMV test time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sequential.gemv(1.0, A, x, 0.0, y, n, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "SEQ VecSum test time: " << elapsed2.count() << " seconds\n";


    double sd = (elapsed1.count()/elapsed2.count()); 
    std::cout << "Speed-down: " << sd  << std::endl; 

 
    for (int i = 0; i < n; i++)
        if(b[i] != n){
            std::cerr << "full gemv not passed\n";
            return;
        }

    std::clog << "full gemv passed\n";

    delete[] b;
}

void vecsum_test(int n) {
    auto *a = new (std::align_val_t{64}) double[n];
    auto *c = new (std::align_val_t{64}) double[n];
    auto *d = new (std::align_val_t{64}) double[n];

    for (int i = 0; i < n; i++)
        a[i] = d[i] = i;

    cl_int err;
    cl::Buffer d_a(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)a, &err);
    cl::Buffer d_b(context, CL_CHANNEL_2_INTELFPGA | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)a, &err);
    CHECK_ERR(err, "err vecsum test");

    auto start = std::chrono::high_resolution_clock::now();
    DirectDeviceHandler device(dev, context, q, program, fblas_program);
    device.vec_sum(1.0, d_a, 2.0, d_b, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "FPG VecSum test time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sequential.axpby(1.0, a, 1.0, d, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "SEQ VecSum test time: " << elapsed2.count() << " seconds\n";


    q.enqueueReadBuffer(d_b, CL_TRUE, 0, n*sizeof(double), c);
    double sd = (elapsed1.count()/elapsed2.count()); 
    std::cout << "Speed-down: " << sd  << std::endl; 

    for (int i = 0; i < n; i++)
        if(c[i] != 3*i){
            std::cerr << "vec sum not passed\n";
            return;
        }

    std::clog << "vec sum passed\n";

    delete[] a;
    delete[] c;
}

void dot_test(int n) {
    auto *a = new (std::align_val_t{64}) double[n];
    auto *c = new (std::align_val_t{64}) double[1];

    for (int i = 0; i < n; i++)
        a[i] = 1.0;

    cl_int err;
    cl::Buffer d_a(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, n*sizeof(double), (void*)a, &err);
    cl::Buffer d_b(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, n*sizeof(double), (void*)a, &err);
    cl::Buffer d_c(context, CL_CHANNEL_1_INTELFPGA | CL_MEM_WRITE_ONLY, sizeof(double));
    CHECK_ERR(err, "err dot test");

    auto start = std::chrono::high_resolution_clock::now();
    DirectDeviceHandler device(dev, context, q, program, fblas_program);
    device.dot(d_a, d_b, d_c, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "FPG Dot test time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sequential.dot(a, a, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "SEQ Dot test time: " << elapsed2.count() << " seconds\n";

    double sd = (elapsed1.count()/elapsed2.count()); 
    std::cout << "Speed-down: " << sd  << std::endl; 

    q.enqueueReadBuffer(d_c, CL_TRUE, 0, sizeof(double), c);
    if(c[0] != n)
        std::cerr << "Dot test not passed\n";
    else
        std::clog << "Dot test passed\n";

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
    int n = std::stoi(n_str[0]);
	for(const auto& n: n_str){
		full_test(n);
        int i = std::stoi(n);
	}
	return 0;
}
