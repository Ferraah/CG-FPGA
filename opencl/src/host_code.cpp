#include "host_code.hpp"



void prepare(cl::Context &context, cl::CommandQueue &q, cl::Program &program){


    cl_int err;
    cl::Platform platform;
    std::vector<cl::Platform> all_platforms;
    std::vector<cl::Device> platform_devices;
    cl::Device device;
    //platform = cl::Platform::get(&err); 

    cl::Platform::get(&all_platforms);
    all_platforms[1].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);

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

    //utils::build_source("src/kernels/kernels.cl", program , context);
    utils::load_binaries("/home/users/u101373/CG-FPGA/opencl/bin/fpga/kernels.fpga.acox.aocx", program , context, device);
    /*
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

    q = sycl::queue(selector);

    auto device = q.get_device();

    std::clog << "FPGA preparation, running on device: "
            << device.get_info<sycl::info::device.name>().c_str()
            << std::endl;
    */
}

void conjugate_gradient(cl::Context &context, cl::CommandQueue &q, cl::Program &program, const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error) 
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
    cl::Event event1, event2;

    Device device(context, q, program);

    cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, size*size*sizeof(double), (void*)A, &err);
    cl::Buffer d_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, size*sizeof(double), (void*)b, &err);
    cl::Buffer d_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, size*sizeof(double), (void*)x, &err);
    cl::Buffer d_d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, size*sizeof(double), (void*)d, &err);
    cl::Buffer d_r(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, size*sizeof(double), (void*)r, &err);

    cl::Buffer d_Ad(context, CL_MEM_READ_WRITE, size*sizeof(double));
    cl::Buffer d_dot1(context, CL_MEM_WRITE_ONLY, 1*sizeof(double));
    cl::Buffer d_dot2(context, CL_MEM_WRITE_ONLY, 1*sizeof(double));

    CHECK_ERR(err, "Buffer allocation error. ");


    double alpha, beta;
    double rr, bb;
    double dot1, dot2;

    device.dot(d_b, d_b, d_dot1, size, event1);
    q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &rr);

    bb = rr;

    int num_iters;

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {

        device.matrix_vector_mul(d_A, d_d, d_Ad, size, event1);
        event1.wait();

        device.dot(d_d, d_r, d_dot1, size, event1);
        device.dot(d_Ad, d_d, d_dot2, size, event2);
        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        q.enqueueReadBuffer(d_dot2, CL_TRUE, 0, sizeof(double), &dot2);

        alpha = dot1 / dot2;

        device.vec_sum(alpha, d_d, 1.0, d_x, size, event1);
        device.vec_sum(-alpha, d_Ad, 1.0, d_r, size, event2);
        event1.wait();
        event2.wait();
        

        device.dot(d_Ad, d_r, d_dot1, size, event1);
        device.dot(d_Ad, d_d, d_dot2, size, event2);
        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        q.enqueueReadBuffer(d_dot2, CL_TRUE, 0, sizeof(double), &dot2);

        beta = dot1 / dot2;

        device.vec_sum(1.0, d_r, -beta, d_d, size, event1);
        event1.wait();

        device.dot(d_r, d_r, d_dot1, size, event1);
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

