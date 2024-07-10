#include "host_code.hpp"
#include <chrono>

//#include "CL/cl_ext.h"

#define CL_CHANNEL_1_INTELFPGA              (1<<16)
#define CL_CHANNEL_2_INTELFPGA              (2<<16)
#define CL_CHANNEL_3_INTELFPGA              (3<<16)
#define CL_CHANNEL_4_INTELFPGA              (4<<16)
#define CL_CHANNEL_5_INTELFPGA              (5<<16)
#define CL_CHANNEL_6_INTELFPGA              (6<<16)
#define CL_CHANNEL_7_INTELFPGA              (7<<16)

//#define EMULATOR
#ifdef EMULATOR
#define PLATFORM_ID 0
#else
#define PLATFORM_ID 2
#endif

void prepare(cl::Context &context, cl::CommandQueue &q, cl::Program &program, cl::Program &fblas_program){

    cl_int err = CL_SUCCESS;
    cl::Platform platform;
    std::vector<cl::Platform> all_platforms;
    std::vector<cl::Device> platform_devices;
    cl::Device device;

    cl::Platform::get(&all_platforms);

    all_platforms[PLATFORM_ID].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);

    CHECK_ERR(err, "Failed to get platform ID");
    device = cl::Device::getDefault(&err);
    std::cout << platform_devices.size();
    device = platform_devices[0];
    CHECK_ERR(err, "Failed to get device ID");

    std::cout << "Using default device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    context = cl::Context(device);
    CHECK_ERR(err, "Failed to create context");

    q = cl::CommandQueue(context, device, 0, &err);

    CHECK_ERR(err, "Failed to create command queue");

#ifdef EMULATOR
    utils::build_source("/home/users/u101373/CG-FPGA/opencl/src/FBGABlas_kernels/fblas_kernels_Direct.cl", fblas_program, context, device);
    //utils::build_source("src/kernels/kernels.cl", program , context, device);
#else
    utils::load_binaries("/home/users/u101373/CG-FPGA/opencl/src/fblas_separated/direct_gemv.aocx", fblas_program, context, device);
    //utils::load_binaries("/home/users/u101373/CG-FPGA/opencl/src/FBGABlas_kernels/dgemv.aocx", fblas_program, context, device);
#endif

}

void conjugate_gradient(cl::Device &dev, cl::Context &context, cl::CommandQueue &q, cl::Program &program, cl::Program &fblas_program, const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error) 
{
    double *r = new (std::align_val_t{ 64 }) double[size];
    double *d = new (std::align_val_t{ 64 }) double[size];

    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        d[i] = b[i];
    }

    cl_int err;

    DirectDeviceHandler device(dev, context, q, program, fblas_program);

    cl::Buffer d_A(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size*size*sizeof(double), (void*)A, &err);
    cl::Buffer d_b(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  size*sizeof(double), (void*)b, &err);
    cl::Buffer d_x(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  size*sizeof(double), (void*)x, &err);
    cl::Buffer d_d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size*sizeof(double), (void*)d, &err);
    cl::Buffer d_r(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size*sizeof(double), (void*)r, &err);

    cl::Buffer d_Ad(context,   CL_MEM_READ_WRITE, size*sizeof(double));
    cl::Buffer d_dot1(context, CL_MEM_WRITE_ONLY, 1*sizeof(double));
    cl::Buffer d_dot2(context, CL_MEM_WRITE_ONLY, 1*sizeof(double)); 

    CHECK_ERR(err, "Buffer allocation error. ");

    double alpha, beta;
    double rr, bb;
    double dot1, dot2;

    device.dot(d_b, d_b, d_dot1, size);
    q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &rr);
    
    bb = rr;

    int num_iters;
    
    auto start = std::chrono::high_resolution_clock::now();

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {

        device.matrix_vector_mul(d_A, d_d, d_Ad, size);

        device.dot(d_d, d_r, d_dot1, size);
        device.dot(d_Ad, d_d, d_dot2, size);
        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        q.enqueueReadBuffer(d_dot2, CL_TRUE, 0, sizeof(double), &dot2);
        alpha = dot1 / dot2;

        device.vec_sum(alpha, d_d, 1.0, d_x, size);
        device.vec_sum(-alpha, d_Ad, 1.0, d_r, size);

        device.dot(d_Ad, d_r, d_dot1, size);
        device.dot(d_Ad, d_d, d_dot2, size);

        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        q.enqueueReadBuffer(d_dot2, CL_TRUE, 0, sizeof(double), &dot2);

        beta = dot1 / dot2;

        device.vec_sum(1.0, d_r, -beta, d_d, size);

        device.dot(d_r, d_r, d_dot1, size);
        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        rr = dot1;

        if(std::sqrt(rr / bb) < rel_error) { break; }
    }

    q.enqueueReadBuffer(d_x, CL_TRUE, 0, size*sizeof(double), x);

    delete[] d;
    delete[] r;

    if(num_iters <= max_iters)
    {
        std::clog << "Converged in " << num_iters << " iterations, relative error is: " << std::sqrt(rr / bb) << std::endl;
    }
    else
    {
        std::clog << "Did not converge in " << num_iters << " iterations, relative error is: " << std::sqrt(rr / bb) << std::endl;
    }
}

