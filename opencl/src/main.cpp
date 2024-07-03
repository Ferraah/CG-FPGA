#include <iostream>
#include <vector>

#include "host_code.hpp"
#include "utils.hpp"
#include "Timer.hpp"
#include "sequential.hpp"

void full_test(std::string n_str){

    const std::string _m_path = "/project/home/p200301/tests/matrix" + n_str + ".bin"; 	
    const std::string _rhs_path  = "/project/home/p200301/tests/rhs" + n_str + ".bin"; 	
    const char *m_path = _m_path.c_str();
    const char *rhs_path = _rhs_path.c_str();


    double *matrix;
    double *vector;
    double *x_fpga, *x_seq;

    size_t n, m ; 
    int max_iter = 10000;
    double res = 1.e-4;

    utils::read_matrix_from_file(m_path , matrix, n, m);
    utils::read_vector_from_file(rhs_path, vector, n);

    std::cout << n << std::endl;
    x_fpga = new (std::align_val_t{64}) double[n];
    x_seq = new (std::align_val_t{64}) double[n];


    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;

    Timer timer;
    Sequential sequential;
    
    std::cout << std::endl;
    timer.start();
    prepare(context, q, program);
    timer.stop();
    timer.print_last_formatted();
    
    std::cout << std::endl;
    timer.start();
    conjugate_gradient(context, q, program, matrix, vector, x_fpga, n, max_iter, res);
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

int main(int argc, char* argv[]) {

    
	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << "n1 n2 .. ni" << std::endl;
		return 1;
	}
	std::vector<std::string> n_str(argv + 1, argv + argc);

    find_opencl();

	for(const auto& n: n_str){
		full_test(n);
	}

	return 0;
}

