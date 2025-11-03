#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace nvcuda;


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENABLE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENABLE_EFFICIENT_SMEM_PTR_CAST 0
#endif

template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15}
  uint const i2s = *_i2s;

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; 

#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    i8s[i] = __vsubss4(i8s[i], 0x02020202);
  }
}

template <int M, int N, int K, int M_blocks, int K_block_size, int N_block_size, int M_block_size>
__global__ void __launch_bounds__(512) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ dtype_transform, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
  
  constexpr int M_per_loop = 32;
  constexpr int N_per_loop = 8;
  constexpr int K_per_loop = 16;
  // constexpr int wmma_K = 32;
  
  constexpr int M_loop_end = (M / (M_blocks * M_per_loop));
  constexpr int n_vals = N_per_loop * M_per_loop;
  constexpr int N_loop_end = N / (N_per_loop * M_block_size);
  constexpr int K_loop_num = (N_block_size * K_block_size) / N_per_loop;
  constexpr int K_loop_end = K/(K_per_loop * K_loop_num);
  // int offset;
  // int dst_idx;
  int base_idx;
  int start;
  int total_elements;
  int thread_id = (threadIdx.y * blockDim.x + threadIdx.x);
  
  __nv_bfloat16 ws_int[1];
  __shared__ __nv_bfloat16 s_shared[(M / (M_blocks))];
  // __shared__ signed char A_shared[M_per_loop * K_per_loop * K_block_size];
  __shared__ int Dump_buff[M_block_size * N_per_loop * M_per_loop];
  __shared__ signed char B_decode[M_block_size * K_per_loop * N_block_size * K_block_size];

  
  wmma::fragment<wmma::matrix_a, M_per_loop, N_per_loop, K_per_loop, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M_per_loop, N_per_loop, K_per_loop, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M_per_loop, N_per_loop, K_per_loop, int> c_frag[M / (M_blocks * M_per_loop)];

  start = blockIdx.x * (M / (M_blocks));
  total_elements = (M / (M_blocks)) / 8;

  #pragma unroll
  for (int mem_addr = thread_id; mem_addr < total_elements; mem_addr += blockDim.x * blockDim.y) {
      reinterpret_cast<int4*>(s_shared)[mem_addr] = reinterpret_cast<int4*>(&s[start])[ + mem_addr];
  }

  // start = (blockIdx.x * (M / (M_blocks))) * (K / K_per_loop);

  ws_int[0] = ws[0];

  int B_shared_start = threadIdx.z * K_per_loop * N_block_size * K_block_size;
  // int A_shared_start = threadIdx.z * N_block_size * K_block_size * M_per_loop;
  int dst_offset = (threadIdx.x + threadIdx.y * K_block_size) * K_per_loop;
  int src_offset = (threadIdx.x * 4) + threadIdx.y * 32;
  int write_mem_idx = (blockIdx.x * (M / M_blocks) * N) + threadIdx.z * N_per_loop * (M / M_blocks);
  int dump_buff_offset = threadIdx.z * N_per_loop * M_per_loop;

  #pragma unroll
  for(int m_0 = 0; m_0 < M_loop_end; ++m_0){
    wmma::fill_fragment(c_frag[m_0], (int)0);
  }

  #pragma unroll
  for (int n_0 = 0; n_0 < N_loop_end; ++n_0){
    #pragma unroll
    for (int k_0 = 0; k_0 < K_loop_end; ++k_0) {
      decode_i2s_to_i8s((int*)(B + 
        ((n_0 * M_block_size + threadIdx.z) * (N_per_loop) * K / 4) + 
        (k_0 * K_loop_num * K_per_loop * N_per_loop / 4) + src_offset
      ), 
      &B_decode[B_shared_start + dst_offset], 16);
              
      #pragma unroll
      for(int k_2_0 = 0; k_2_0 < K_loop_num; ++k_2_0){
        wmma::load_matrix_sync(b_frag, &B_decode[B_shared_start + k_2_0 * K_per_loop * N_per_loop], K_per_loop);
        # pragma unroll
        for(int m_0 = 0; m_0 < M_loop_end; ++m_0){
          start = (blockIdx.x * (M / (M_blocks))) * (K) + 
                  k_0 * M_per_loop * K_per_loop * K_loop_num + 
                  m_0 * M_per_loop * K;
          // wmma::load_matrix_sync(a_frag, &A_shared[k_2_0 * M_per_loop * K_per_loop], 16);
          wmma::load_matrix_sync(a_frag, &A[start + k_2_0 * M_per_loop * K_per_loop], K_per_loop);
          wmma::mma_sync(c_frag[m_0], a_frag, b_frag, c_frag[m_0]);
        }
      }
    }

    base_idx = write_mem_idx + n_0 * M_block_size  * N_per_loop * (M / M_blocks);

    #pragma unroll
    for(int m_0 = 0; m_0 < M_loop_end; ++m_0){
      
      wmma::store_matrix_sync(&Dump_buff[dump_buff_offset], c_frag[m_0], N_per_loop, wmma::mem_row_major);

      #pragma unroll
      for(int idx = thread_id; idx < n_vals; idx = idx + N_block_size * K_block_size){
        dtype_transform[base_idx + m_0 * n_vals + idx] =
          ((__nv_bfloat16)(Dump_buff[dump_buff_offset + idx]) /
          (s_shared[m_0 * M_per_loop + (idx / N_per_loop)] *
          ws_int[0])
        ); 
      }
      wmma::fill_fragment(c_frag[m_0], (int)0);
    }
  }
}
