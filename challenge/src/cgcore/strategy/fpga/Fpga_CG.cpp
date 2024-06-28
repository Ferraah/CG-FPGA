#include "Fpga_CG.hpp"

#include <stdexcept>
#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>
#include <chrono>

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class DotProduct;
class GEMV;
class VecSum;

constexpr uint II_CYCLES = 64;

// Function to demangle C++ function names
std::string demangle(const char* symbol) {
    size_t size;
    int status;
    char* demangled = abi::__cxa_demangle(symbol, nullptr, &size, &status);
    std::string result(symbol);
    if (status == 0) {
        result = std::string(demangled);
        std::free(demangled);
    }
    return result;
}

// Function to print the stack trace
void print_stack_trace() {
    const int max_frames = 128;
    void* addrlist[max_frames + 1];

    // Retrieve current stack addresses
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

    if (addrlen == 0) {
        std::cerr << "No stack trace available\n";
        return;
    }

    // Create readable strings to each frame.
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    // Iterate over each frame and demangle if possible
    for (int i = 0; i < addrlen; i++) {
        std::string symbol(symbollist[i]);

        // Find and demangle the mangled function name
        size_t begin = symbol.find('(');
        size_t end = symbol.find('+', begin);
        if (begin != std::string::npos && end != std::string::npos) {
            std::string mangled_name = symbol.substr(begin + 1, end - begin - 1);
            std::string demangled_name = demangle(mangled_name.c_str());
            symbol.replace(begin + 1, end - begin - 1, demangled_name);
        }

        std::cerr << symbol << std::endl;
    }

    std::free(symbollist);
}

namespace cgcore{
    
    void FPGA_CG::dot(sycl::queue &q, const double *dA, const double *dB, double *dC, size_t size) const 
    {
        q.submit([&](sycl::handler &h){
            h.single_task<DotProduct>([=](){
                double shift_reg[II_CYCLES+1];

                for (int i = 0; i < II_CYCLES + 1; i++) {
                    shift_reg[i] = 0;
                }

                for(int i=0; i< size; i++){
                    shift_reg[II_CYCLES] = shift_reg[0] + dA[i]*dB[i];

                    #pragma unroll
                    for(int j=0;j<II_CYCLES;j++){
                        shift_reg[j]=shift_reg[j+1];
                    }
                }

                double temp_sum = 0; 
                #pragma unroll
                for (int i = 0; i < II_CYCLES; i++) {
                    temp_sum += shift_reg[i];
                }

                dC[0] = temp_sum;
            });
        });
    }

    void FPGA_CG::vec_sum(sycl::queue &q,double alpha, const double *dX, double beta, double *dY, size_t size) const
    {
        q.submit([&](sycl::handler &h){
            h.single_task<VecSum>([=](){
                const int aux_dim = II_CYCLES + 1;
                double aux_reg[aux_dim];
                
                for (int i = 0; i < aux_dim; i++)
                {
                    aux_reg[i] = 0;
                }

                int n_cyles = (size + aux_dim - 1) / (aux_dim);
                int base;

                for (int i = 0; i < n_cyles; ++i)
                {
                    base = i*aux_dim; 

                    for (int j = 0; j < aux_dim; ++j)
                    {
                        if (base + j < size)
                            aux_reg[j] =  alpha * dX[base + j] + beta * dY[base + j];
                        else 
                            aux_reg[j] = 0;
                    }

                    for (int j = 0; j < aux_dim; ++j)
                    {
                        if (base + j < size)
                            dY[base + j] = aux_reg[j];
                    }
                }
            });
        });
    }

    void FPGA_CG::matrix_vector_mul(sycl::queue &q,const double *dA, const double *dB, double *dC, size_t size) const
    {
        q.submit([&](sycl::handler &h){
            h.single_task<GEMV>([=](){
                double shift_reg[II_CYCLES+1];

                for(size_t i = 0; i < size; i++){

                    for (int k = 0; k < II_CYCLES + 1; k++) {
                        shift_reg[k] = 0;
                    }

                    for(int j=0; j < size; j++){
                        shift_reg[II_CYCLES] = shift_reg[0] + dA[i*size + j]*dB[j];

                        #pragma unroll
                        for(int k=0;k<II_CYCLES;k++){
                            shift_reg[k]=shift_reg[k+1];
                        }
                    }

                    double temp_sum = 0; 
                    #pragma unroll
                    for (int k = 0; k < II_CYCLES; k++) {
                        temp_sum += shift_reg[k];
                    }

                    dC[i] = temp_sum; 
                }
            });
        });
    }

    void FPGA_CG::prepare(){

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

        q = sycl::queue(selector, sycl::property::queue::in_order{});

        auto device = q.get_device();

        std::clog << "FPGA preparation, running on device: "
                << device.get_info<sycl::info::device::name>().c_str()
                << std::endl;
    }
    
    void FPGA_CG::conjugate_gradient(const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error) 
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
                //auto start = std::chrono::high_resolution_clock::now();
                q.memcpy(d_A, A, size*size*sizeof(double)).wait_and_throw();
                q.memcpy(d_b, b, size*sizeof(double)).wait_and_throw();
                q.memcpy(d_d, d, size*sizeof(double)).wait_and_throw();
                q.memcpy(d_r, r, size*sizeof(double)).wait_and_throw();
                q.memcpy(d_x, x, size*sizeof(double)).wait_and_throw();

                //auto end = std::chrono::high_resolution_clock::now();
                //std::chrono::duration<double> duration = end - start;
                //std::clog << "Initial mem copies duration: " << duration.count() << " seconds" << std::endl;
            }
            catch (sycl::exception const &e) {
                std::cerr << "FPGA_CG > Initial mem_copies error:\n" << e.what() << "\n";
                throw(e);
            }

            try{
                //auto start = std::chrono::high_resolution_clock::now();
                
                dot(q, d_b, d_b, d_dot1, size);
                q.wait_and_throw();
                q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();

                //auto end = std::chrono::high_resolution_clock::now();
                //std::chrono::duration<double> duration = end - start;
                //std::clog << "First dot product duration: " << duration.count() << " seconds" << std::endl;
            }
            catch (sycl::exception const &e) {
                std::cerr << "FPGA_CG > First dot product or its mem copy error:\n" << e.what() << "\n";
                throw(e);
            }

            bb = h_dot1[0];

            for(num_iters = 1; num_iters <= max_iters; num_iters++)
            {
                //auto start = std::chrono::high_resolution_clock::now();

                matrix_vector_mul(q, d_A, d_d, d_Ad, size);
                q.wait_and_throw();

                dot(q, d_d, d_r, d_dot1, size);
                dot(q, d_Ad, d_d, d_dot2, size);
                q.wait_and_throw();

                q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();
                q.memcpy(h_dot2, d_dot2, 1*sizeof(double)).wait_and_throw();
                alpha = h_dot1[0] / h_dot2[0];

                vec_sum(q, alpha, d_d, 1.0, d_x, size);
                vec_sum(q, -alpha, d_Ad, 1.0, d_r, size);
                q.wait_and_throw();

                dot(q, d_Ad, d_r, d_dot1, size);
                dot(q, d_Ad, d_d, d_dot2, size);
                q.wait_and_throw();

                q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();
                q.memcpy(h_dot2, d_dot2, 1*sizeof(double)).wait_and_throw();
                beta = h_dot1[0] / h_dot2[0];

                vec_sum(q, 1.0, d_r, -beta, d_d, size);
                q.wait_and_throw();

                dot(q, d_r, d_r, d_dot1, size);
                q.wait_and_throw();
                q.memcpy(h_dot1, d_dot1, 1*sizeof(double)).wait_and_throw();
                rr = h_dot1[0]; 

                //auto end = std::chrono::high_resolution_clock::now();
                //std::chrono::duration<double> duration = end - start;
                //std::clog << "Iteration " << num_iters << " duration: " << duration.count() << " seconds" << std::endl;

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
            print_stack_trace();

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

    void FPGA_CG::run(const double *A, const double *b, double *x, size_t size, int max_iter, double res_error)
    {
        conjugate_gradient(A, b, x, size, max_iter, res_error);
    }
}
