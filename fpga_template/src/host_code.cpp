#include "host_code.hpp"
// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class DotProduct1;
class DotProduct2;
class DotProduct3;
class DotProduct4;
class DotProduct5;
class DotProduct6;
class GEMV;
class VecSum1;
class VecSum2;
class VecSum3;

void prepare(sycl::queue &q){

    #if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
    #elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
    #else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    #endif

    q = sycl::queue(selector);

    auto device = q.get_device();

    std::clog << "FPGA preparation, running on device: "
            << device.get_info<sycl::info::device::name>().c_str()
            << std::endl;
}

void conjugate_gradient(sycl::queue &q, const double *h_A, const double *h_b, double *h_x, size_t size, int max_iters, double rel_error) 
{

    try{
        // Host memory --------------------------------------- 
        double* h_Ad = new (std::align_val_t{ 64 }) double[size];
        double* h_d  = new (std::align_val_t{ 64 }) double[size];
        double* h_r  = new (std::align_val_t{ 64 }) double[size];

        // dot product results
        double* h_dot1 = new (std::align_val_t{ 64 }) double[1];
        double* h_dot2 = new (std::align_val_t{ 64 }) double[1];
        // ----------------------------------------------------

        double alpha, beta;
        double rr, bb;  // To check relative error

        int num_iters;

        // Buffers ---------------------------------------------

        sycl::buffer<double> buffer_d_A  (&h_A[0], size*size);
        sycl::buffer<double> buffer_d_Ad (&h_Ad[0], size);
        sycl::buffer<double> buffer_d_d  (&h_d[0], size);
        sycl::buffer<double> buffer_d_r  (&h_r[0], size);
        sycl::buffer<double> buffer_d_x  (&h_x[0], size);
        sycl::buffer<double> buffer_d_b  (&h_b[0], size);
        
        sycl::buffer<double> buffer_d_dot1  (&h_dot1[0], 1);
        sycl::buffer<double> buffer_d_dot2 (&h_dot2[0], 1);


        sycl::buffer<double> buffer_r (&h_r[0], size);
        sycl::buffer<double> buffer_d (&h_d[0], size);
        // -----------------------------------------------------

        // Initializing the vectors with host accessors --------
        {        
            sycl::host_accessor x_acc(buffer_d_x);
            sycl::host_accessor r_acc(buffer_d_r);
            sycl::host_accessor d_acc(buffer_d_d);
            sycl::host_accessor b_acc(buffer_d_b);

            for(size_t i = 0; i < size; i++)
            {
                x_acc[i] = 0.0;
                r_acc[i] = b_acc[i];
                d_acc[i] = b_acc[i];

            }

        }

        q.submit([&](sycl::handler &h)
        {
            sycl::accessor acc_d (buffer_d_d, h, sycl::read_only);
            sycl::accessor acc_b (buffer_d_b, h, sycl::read_only);
            sycl::accessor acc_dot1 (buffer_d_dot1, h, sycl::write_only);

            h.single_task<DotProduct1>([=](){
                Device::dot(&acc_d[0], &acc_b[0], &acc_dot1[0], size);
            });
        });

        {
            sycl::host_accessor dot1_acc(buffer_d_dot1);
            bb = dot1_acc[0];
        }


        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            // Calculating A*d
            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_A (buffer_d_A, h, sycl::read_only);
                sycl::accessor acc_d (buffer_d_d, h, sycl::read_only);
                sycl::accessor acc_Ad (buffer_d_Ad, h, sycl::write_only);

                h.single_task<GEMV>([=](){
                    Device::matrix_vector_mul(&acc_A[0], &acc_d[0], &acc_Ad[0], size);
                });
            });

            // Calculating alpha = d*r / (Ad * d) in parallel
            
            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_d (buffer_d_d, h, sycl::read_only);
                sycl::accessor acc_r (buffer_d_r, h, sycl::read_only);
                sycl::accessor acc_dot1 (buffer_d_dot1, h, sycl::write_only);

                h.single_task<DotProduct2>([=](){
                    Device::dot(&acc_d[0], &acc_r[0], &acc_dot1[0], size);
                });
            });

            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_Ad (buffer_d_Ad, h, sycl::read_only);
                sycl::accessor acc_d (buffer_d_d, h, sycl::read_only);
                sycl::accessor acc_dot2 (buffer_d_dot2, h, sycl::write_only);

                h.single_task<DotProduct3>([=](){
                    Device::dot(&acc_Ad[0], &acc_d[0], &acc_dot2[0], size);
                });
            });

            // dot(q, d_d, d_r, d_dot1, size);
            // dot(q, d_Ad, d_d, d_dot2, size);
            {
                sycl::host_accessor d_dot1_acc(buffer_d_dot1);
                sycl::host_accessor d_dot2_acc(buffer_d_dot2);
                alpha =  d_dot1_acc[0]/d_dot2_acc[0];
            }

            // Updating x along d and r

            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_d (buffer_d_d, h, sycl::read_only);
                sycl::accessor acc_x (buffer_d_x, h, sycl::read_write);

                h.single_task<VecSum1>([=]()
                {
                    Device::vec_sum(alpha, &acc_d[0], 1.0, &acc_x[0], size);
                });
            });

            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_Ad (buffer_d_Ad, h, sycl::read_only);
                sycl::accessor acc_r (buffer_d_r, h, sycl::read_write);

                h.single_task<VecSum2>([=]()
                {
                    Device::vec_sum(-alpha, &acc_Ad[0], 1.0, &acc_r[0], size);
                });
            });

            // vec_sum(q, alpha, d_d, 1.0, d_x, size);
            // vec_sum(q, -alpha, d_Ad, 1.0, d_r, size);

            // Calculating Beta
            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_Ad (buffer_d_Ad, h, sycl::read_only);
                sycl::accessor acc_r (buffer_d_r, h, sycl::read_only);
                sycl::accessor acc_dot1 (buffer_d_dot1, h, sycl::write_only);

                h.single_task<DotProduct4>([=](){
                    Device::dot(&acc_Ad[0], &acc_r[0], &acc_dot1[0], size);
                });
            });

            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_Ad (buffer_d_Ad, h, sycl::read_only);
                sycl::accessor acc_d (buffer_d_d, h, sycl::read_only);
                sycl::accessor acc_dot2 (buffer_d_dot2, h, sycl::write_only);

                h.single_task<DotProduct5>([=](){
                    Device::dot(&acc_Ad[0], &acc_d[0], &acc_dot2[0], size);
                });
            });


            // dot(q, d_Ad, d_r, d_dot1, size);
            // dot(q, d_Ad, d_d, d_dot2, size);

            {
                sycl::host_accessor d_dot1_acc(buffer_d_dot1);
                sycl::host_accessor d_dot2_acc(buffer_d_dot2);
                beta =  d_dot1_acc[0]/d_dot2_acc[0];
            }

            // Updating d

            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_r (buffer_d_r, h, sycl::read_only);
                sycl::accessor acc_d (buffer_d_d, h, sycl::read_write);

                h.single_task<VecSum3>([=]()
                {
                    Device::vec_sum(1.0, &acc_r[0], -beta, &acc_d[0], size);
                });
            });

            // vec_sum(q, 1.0, d_r, -beta, d_d, size);

            // Checking residual conditions

            q.submit([&](sycl::handler &h)
            {
                sycl::accessor acc_r (buffer_d_r, h, sycl::read_only);
                sycl::accessor acc_dot1 (buffer_d_dot1, h, sycl::write_only);

                h.single_task<DotProduct6>([=](){
                    Device::dot(&acc_r[0], &acc_r[0], &acc_dot1[0], size);
                });
            });

            // dot(q, d_r, d_r, d_dot1, size);

            {
                sycl::host_accessor d_dot1_acc (buffer_d_dot1);
                rr = d_dot1_acc[0]; 
            }

            if(std::sqrt(rr / bb) < rel_error) { break; }
        }

        if(num_iters <= max_iters)
        {
            std::clog << "Converged in " << num_iters << " iterations, relative error is: " << std::sqrt(rr / bb) << std::endl;
        }
        else
        {
            std::clog << "Did not converge in " << num_iters << " iterations, relative error is: " << std::sqrt(rr / bb) << std::endl;
        }

        delete[] h_Ad;
        delete[] h_d;
        delete[] h_r;
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

