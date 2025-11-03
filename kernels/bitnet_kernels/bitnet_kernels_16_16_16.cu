#include "bitnet_kernels_16_16_16.h"

extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K, cudaStream_t stream){
    if(M == 1024 && N == 512 && K == 2048){
        ladder_int8xint2_kernel<1024, 512, 2048, 32, 8, 4, 16><<<dim3(32, 1, 1), dim3(8, 4, 16), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1024 && N == 2048 && K == 512){
        ladder_int8xint2_kernel<1024, 2048, 512, 32, 8, 4, 16><<<dim3(32, 1, 1), dim3(8, 4, 16), 0, stream>>>(input0, input1, output0, s, ws);
    } 
    // else if(M == 1 && N == 512 && K == 2048){
    //     ladder_int8xint2_kernel<1, 512, 2048, 1, 1, 8, 16><<<dim3(32, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    // }
    // else if(M == 1 && N == 2048 && K == 512){
    //     ladder_int8xint2_kernel<1, 2048, 512, 1, 1, 8, 16><<<dim3(128, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    // }     
    else{
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}