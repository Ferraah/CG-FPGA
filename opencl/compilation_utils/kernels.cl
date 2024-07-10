#define II_CYCLES 64
#define UNROLL 4

__kernel void DotProduct(__global const double* restrict dA, __global const double* restrict dB, __global double* restrict dC, const unsigned size) {
    double shift_reg[II_CYCLES+1];

    #pragma unroll 
    for (int i = 0; i < II_CYCLES + 1; i++) {
        shift_reg[i] = 0;
    }
    
    for (int i = 0; i < size; i++) {
        shift_reg[II_CYCLES] = shift_reg[0] + dA[i] * dB[i];
        
        #pragma unroll
        for (int j = 0; j < II_CYCLES; j++) {
            shift_reg[j] = shift_reg[j+1];
        }
    }
    
    double temp_sum = 0;
    #pragma unroll
    for (int i = 0; i < II_CYCLES; i++) {
        temp_sum += shift_reg[i];
    }
    
    dC[0] = temp_sum;
}

__kernel void VecSum(const double alpha, __global const double* restrict dX, const double beta, __global double* restrict dY, const unsigned size) {

    const int aux_dim = II_CYCLES + 1;
    double aux_reg[aux_dim];
    
    for (int i = 0; i < aux_dim; i++) {
        aux_reg[i] = 0;
    }
    
    int n_cyles = (size + aux_dim - 1) / aux_dim;
    int base;
    
    for (int i = 0; i < n_cyles; ++i) {
        base = i * aux_dim;
        
        for (int j = 0; j < aux_dim; ++j) {
            if (base + j < size)
                aux_reg[j] = alpha * dX[base + j] + beta * dY[base + j];
            else
                aux_reg[j] = 0;
        }
        
        for (int j = 0; j < aux_dim; ++j) {
            if (base + j < size)
                dY[base + j] = aux_reg[j];
        }
    }
}

__kernel void GEMV(__global const double* restrict dA, __global const double* restrict dB, __global double* restrict dC, const unsigned size) 
{

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
}


