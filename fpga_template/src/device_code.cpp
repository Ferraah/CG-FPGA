#include "device_code.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class DotProduct;
class GEMV;
class VecSum;

constexpr uint II_CYCLES = 16;
    
void Device::dot(sycl::queue &q, const double *dA, const double *dB, double *dC, size_t size) 
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

void Device::vec_sum(sycl::queue &q,double alpha, const double *dX, double beta, double *dY, size_t size) 
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

void Device::matrix_vector_mul(sycl::queue &q,const double *dA, const double *dB, double *dC, size_t size) 
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



