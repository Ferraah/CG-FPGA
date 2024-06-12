#include "Fpga_CG.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class DotProduct;
class GEMV;
class VecSum;


constexpr uint II_CYCLES = 256; 

namespace cgcore{
    
    /**
     * @param dA The reference to the first vector in device memory 
     * @param dB The reference to the first vector in device memory 
     * @param dC The reference to the result in device memory
     * @param size The size of the vectors 
    */
    void FPGA_CG::dot(sycl::queue &q, const double *dA, const double *dB, double *dC,  size_t size) const 
    {
        q.submit([&](sycl::handler &h){
            h.single_task<DotProduct>([=](){

                //Create shift register with II_CYCLE+1 elements
                double shift_reg[II_CYCLES+1];

                //Initialize all elements of the register to 0
                //You must initialize the shift register 
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

    /**
     * Sum two vector and replace the result as the first addend.
     * @param alpha
     * @param dX Reference to first vector in device memory
     * @param beta 
     * @param dY Reference to second vector in device memory
     * @param size Size of the two vectors
    */
    void FPGA_CG::vec_sum(sycl::queue &q,double alpha, const double *dX, double beta, double *dY, size_t size) const
    {
        q.submit([&](sycl::handler &h){
            h.single_task<VecSum>([=](){


                double res_idx;
                for (size_t idx = 0; idx < size; idx++) {
                    res_idx = alpha * dX[idx] + beta * dY[idx];
                    dY[idx] = res_idx;
                }

            });
        });

    }

    /**
     * Run the matrix vector multiplication on the selected device. 
     * @param kernel the kernel associated to the method
     * @param dA the device pointer to the matrix A, already loaded in device memory. 
     * @param dB the device pointer to the vector b, already loaded in device memory.
     * @param dC the device pointer to the result c, in device memory.
     * 
     * @TODO: Remove A parameter
    */
    void FPGA_CG::matrix_vector_mul(sycl::queue &q,const double *dA, const double *dB, double *dC, size_t size) const
    {
        q.submit([&](sycl::handler &h){
            h.single_task<GEMV>([=](){
                
                //Create shift register with II_CYCLE+1 elements
                double shift_reg[II_CYCLES+1];

                // For every row
                for(size_t i = 0; i < size; i++){

                    /*
                    #pragma unroll
                    for(size_t j = 0; j < size; j++){
                        sum += dA[i * size + j] * dB[j];
                    }
                    */

                    // DOT PRODUCT
                    {
                        // Initialize all elements of the register to 0
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
                        //
                    }
                }
            });
        });

    }

    FPGA_CG::FPGA_CG(){
   }

    /**
     * Main method for the strategy 
    */ 
    void FPGA_CG::conjugate_gradient(const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error) const
    {
        // Use compile-time macros to select either:
        //  - the FPGA emulator device (CPU emulation of the FPGA)
        //  - the FPGA device (a real FPGA)
        //  - the simulator device
        #if FPGA_SIMULATOR
        auto selector = sycl::ext::intel::fpga_simulator_selector_v;
        #elif FPGA_HARDWARE
        auto selector = sycl::ext::intel::fpga_selector_v;
        #else  // #if FPGA_EMULATOR
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
        #endif

        // create the device queue
        sycl::queue q(selector, sycl::property::queue::in_order{} );
        // make sure the device supports USM host allocations
        auto device = q.get_device();

        std::cout << "Running on device: "
                  << device.get_info<sycl::info::device::name>().c_str()
                  << std::endl;
 
        // Device memory --------------------------------------- 
        auto d_A  = sycl::malloc_device<double>(size*size, q);
        auto d_Ad = sycl::malloc_device<double>(size, q);
        auto d_d  = sycl::malloc_device<double>(size, q);
        auto d_r  = sycl::malloc_device<double>(size, q);
        auto d_x  = sycl::malloc_device<double>(size, q);
        auto d_b  = sycl::malloc_device<double>(size, q);
        // dot product results
        auto d_dot1 = sycl::malloc_shared<double>(1, q);
        auto d_dot2 = sycl::malloc_shared<double>(1, q);
        // ----------------------------------------------------

        double alpha, beta;
        double rr, bb;  // To check relative error

        // Starting conditions
        double *r = new double[size];
        double *d = new double[size];

        int num_iters;

        for(size_t i = 0; i < size; i++)
        {
            x[i] = 0.0;
            r[i] = b[i];
            d[i] = b[i];
        }

        // Copy initial data to device  
        q.memcpy(d_A, A, size*size*sizeof(double)).wait();
        q.memcpy(d_b, b, size*sizeof(double)).wait();
        q.memcpy(d_d, d, size*sizeof(double)).wait();
        q.memcpy(d_r, r, size*sizeof(double)).wait();
        q.memcpy(d_x, x, size*sizeof(double)).wait();

        dot(q, d_b, d_b, d_dot1, size);
        q.wait();
        bb = d_dot1[0];

        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            // Calculating A*d
            matrix_vector_mul(q, d_A, d_d, d_Ad, size);
            q.wait();

            // Calculating alpha = d*r / (Ad * d) in parallel
            dot(q, d_d, d_r, d_dot1, size);
            dot(q, d_Ad, d_d, d_dot2, size);
            q.wait();

            alpha =  d_dot1[0]/d_dot2[0];

            // Updating x along d and r
            vec_sum(q, alpha, d_d, 1.0, d_x, size);
            vec_sum(q, -alpha, d_Ad, 1.0, d_r, size);
            q.wait();

            // Calculating Beta
            dot(q, d_Ad, d_r, d_dot1, size);
            dot(q, d_Ad, d_d, d_dot2, size);
            q.wait();

            beta =  d_dot1[0]/d_dot2[0];

            // Updating d
            vec_sum(q, 1.0, d_r, -beta, d_d, size);
            q.wait();

            // Checking residual conditions
            dot(q, d_r, d_r, d_dot1, size);
            q.wait();
            rr = d_dot1[0]; 

            if(std::sqrt(rr / bb) < rel_error) { break; }
        }

        q.memcpy(x, d_x, size*sizeof(double)).wait();

        delete[] d;
        delete[] r;

        if(num_iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }

        sycl::free(d_A, q);
        sycl::free(d_Ad, q);
        sycl::free(d_d, q);
        sycl::free(d_r, q);
        sycl::free(d_x, q);
        sycl::free(d_b, q);
    }

    /**
     * Common strategy interface.
    */
    void FPGA_CG::run(const double *A, const double *b, double *x, size_t size, int max_iter, double res_error) const 
    {
        conjugate_gradient(A, b, x, size, max_iter, res_error);
    }
}
