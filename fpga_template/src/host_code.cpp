#include "host_code.hpp"

void prepare(sycl::queue &q){

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
            << device.get_info<sycl::info::device::name>().c_str()
            << std::endl;
}

void conjugate_gradient(sycl::queue &q, const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error) 
{

    try{


        double* d_A;
        double* d_Ad;
        double* d_d;
        double* d_r;
        double* d_x;
        double* d_b;
        double* d_dot1;
        double* d_dot2;

        try{

            d_A  = sycl::malloc_device<double>(size*size, q);
            d_Ad = sycl::malloc_device<double>(size, q);
            d_d  = sycl::malloc_device<double>(size, q);
            d_r  = sycl::malloc_device<double>(size, q);
            d_x  = sycl::malloc_device<double>(size, q);
            d_b  = sycl::malloc_device<double>(size, q);
            d_dot1 = sycl::malloc_device<double>(1, q);
            d_dot2 = sycl::malloc_device<double>(1, q);
            q.wait_and_throw();

        }
        catch (sycl::exception const &e) {
            std::cerr << "FPGA_CG > Device malloc errors:\n" << e.what() << "\n";
            throw(e);
        }

        double* h_dot1 = new (std::align_val_t{ 64 }) double[1];
        double* h_dot2 = new (std::align_val_t{ 64 }) double[1];

        double alpha, beta;
        double rr, bb;

        double *r = new (std::align_val_t{ 64 }) double[size];
        double *d = new (std::align_val_t{ 64 }) double[size];

        int num_iters;

        for(size_t i = 0; i < size; i++)
        {
            x[i] = 0.0;
            r[i] = b[i];
            d[i] = b[i];
        }

        try{    

            q.memcpy(d_A, A, size*size*sizeof(double)).wait_and_throw();
            q.memcpy(d_b, b, size*sizeof(double)).wait_and_throw();
            q.memcpy(d_d, d, size*sizeof(double)).wait_and_throw();
            q.memcpy(d_r, r, size*sizeof(double)).wait_and_throw();
            q.memcpy(d_x, x, size*sizeof(double)).wait_and_throw();

        }
        catch (sycl::exception const &e) {
            std::cerr << "FPGA_CG > Initial mem_copies error:\n" << e.what() << "\n";
            throw(e);
        }

        try{

            Device::dot(q, d_b, d_b, d_dot1, size);
            q.wait_and_throw();
            q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();

        }
        catch (sycl::exception const &e) {
            std::cerr << "FPGA_CG > First dot product or its mem copy error:\n" << e.what() << "\n";
            throw(e);
        }

        bb = h_dot1[0];

        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {

            Device::matrix_vector_mul(q, d_A, d_d, d_Ad, size);
            q.wait_and_throw();

            Device::dot(q, d_d, d_r, d_dot1, size);
            Device::dot(q, d_Ad, d_d, d_dot2, size);
            q.wait_and_throw();

            q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();
            q.memcpy(h_dot2, d_dot2, 1*sizeof(double)).wait_and_throw();
            alpha = h_dot1[0] / h_dot2[0];

            Device::vec_sum(q, alpha, d_d, 1.0, d_x, size);
            Device::vec_sum(q, -alpha, d_Ad, 1.0, d_r, size);
            q.wait_and_throw();

            Device::dot(q, d_Ad, d_r, d_dot1, size);
            Device::dot(q, d_Ad, d_d, d_dot2, size);
            q.wait_and_throw();

            q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();
            q.memcpy(h_dot2, d_dot2, 1*sizeof(double)).wait_and_throw();
            beta = h_dot1[0] / h_dot2[0];

            Device::vec_sum(q, 1.0, d_r, -beta, d_d, size);
            q.wait_and_throw();

            Device::dot(q, d_r, d_r, d_dot1, size);
            q.wait_and_throw();
            q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();
            rr = h_dot1[0]; 

            if(std::sqrt(rr / bb) < rel_error) { break; }
        }

        q.memcpy(x, d_x, size*sizeof(double)).wait_and_throw();

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

        sycl::free(d_A, q);
        sycl::free(d_Ad, q);
        sycl::free(d_d, q);
        sycl::free(d_r, q);
        sycl::free(d_x, q);
        sycl::free(d_b, q);
        sycl::free(d_dot1, q);
        sycl::free(d_dot2, q);

        delete[] h_dot1;
        delete[] h_dot2;
    }
    catch (sycl::exception const &e) {
        std::cerr << "FPGA_CG > Caught a synchronous SYCL host exception:\n" << e.what() << "\n";
        //print_stack_trace();

        if (e.code().value() == CL_DEVICE_NOT_FOUND) {
            std::cerr << "If you are targeting an FPGA, please ensure that your "
                        "system has a correctly configured FPGA board.\n";
            std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
            std::cerr << "If you are targeting the FPGA emulator, compile with "
                        "-DFPGA_EMULATOR.\n";
        }

        std::terminate();
    }
}

