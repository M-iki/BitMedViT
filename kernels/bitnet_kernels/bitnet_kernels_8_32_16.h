#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>

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
  // uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15}
  // uint i2s = *_i2s;

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; 

#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(reinterpret_cast<uint *>(_i8s)[i])
                 : "r"(*_i2s), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    *_i2s >>= 2;
    reinterpret_cast<uint *>(_i8s)[i] = __vsubss4(reinterpret_cast<uint *>(_i8s)[i], 0x02020202);
  }
}

template <int M, int N, int K, int N_blocks, int M_blocks, int K_block_size, int N_block_size, int M_block_size>
__global__ void __launch_bounds__(512) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ dtype_transform, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
  
  constexpr int K_per_loop = 16;
  constexpr int M_per_loop = 8;
  constexpr int N_per_loop = 32;
  
  constexpr int M_loop_end = (M / (M_blocks * M_per_loop));
  constexpr int n_vals = N_per_loop * M_per_loop;
  constexpr int N_loop_end = N / (N_per_loop * M_block_size * N_blocks);
  constexpr int K_loop_num = (N_block_size * K_block_size) / N_per_loop;
  constexpr int K_loop_end = K/(K_per_loop * K_loop_num);
  // constexpr int buf_size = 2;
  const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
  const cooperative_groups::coalesced_group warp = cooperative_groups::coalesced_threads();
  // int offset;
  // int dst_idx;
  int base_idx;
  int start;
  int total_elements;
  int thread_id = (threadIdx.y * blockDim.x + threadIdx.x);
  
  __nv_bfloat16 ws_int[1];
  __shared__ __nv_bfloat16 s_shared[(M / (M_blocks))];
  __shared__ signed char A_shared[M_block_size * (M / M_per_loop) * K_per_loop];
  // __shared__ signed char A_shared[M_block_size * buf_size * (M / M_per_loop) * K_block_size];
  __shared__ int Dump_buff[M_block_size * N_per_loop * M_per_loop];
  // __shared__ int B_shared_buf[buf_size * M_block_size * N_block_size * K_block_size];
  __shared__ signed char B_decode[M_block_size * K_per_loop * N_block_size * K_block_size];
  signed char B_decode_local[K_per_loop];
  int B_local[1];
    

  wmma::fragment<wmma::matrix_a, M_per_loop, N_per_loop, K_per_loop, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M_per_loop, N_per_loop, K_per_loop, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M_per_loop, N_per_loop, K_per_loop, int> c_frag[M / (M_blocks * M_per_loop)];

  ws_int[0] = ws[0];
  
  start = blockIdx.y * (M / (M_blocks));
  total_elements = (M / (M_blocks));

  #pragma unroll
  for (int mem_addr = thread_id + threadIdx.z * N_block_size * K_block_size; mem_addr < total_elements; mem_addr += blockDim.x * blockDim.y * blockDim.z) {
      s_shared[mem_addr] = ((s[start + mem_addr])  * ws_int[0]);
  }
  __syncthreads();
  
  // start = (blockIdx.x * (M / (M_blocks))) * (K / K_per_loop);


  int B_shared_start = threadIdx.z * K_per_loop * N_block_size * K_block_size;
  // int A_shared_start = threadIdx.z * N_block_size * K_block_size * M_per_loop;
  int dst_offset = (threadIdx.x + (threadIdx.y & 1) * 16 + ((threadIdx.y >> 1) & 1) * 8) * K_per_loop;
  int src_offset = (threadIdx.x * 4) + threadIdx.y * 32;
  int write_mem_idx = (blockIdx.y * (M / M_blocks) * N) + threadIdx.z * N_per_loop;
  int dump_buff_offset = threadIdx.z * N_per_loop * M_per_loop;
  // int buf_idx = 0;
  // total_elements = M_block_size * (M / (M_blocks)) * K_per_loop / K_per_loop;
  total_elements = M_block_size * (M / (M_blocks)) * K_per_loop / K_per_loop;

  #pragma unroll
  for(int m_0 = 0; m_0 < M_loop_end; ++m_0){
    wmma::fill_fragment(c_frag[m_0], (int)0);
  }

  #pragma unroll
  for (int n_0 = 0; n_0 < N_loop_end; ++n_0){    
    
    // B_shared_buf[(threadIdx.z * buf_size * N_block_size * K_block_size + threadIdx.y * K_block_size + threadIdx.x + buf_idx * N_block_size * K_block_size)] = *(int*)(B + 
    //   (((n_0 * N_blocks + blockIdx.x) * M_block_size + threadIdx.z) * (N_per_loop) * K / 4) + src_offset);
    // start = (blockIdx.y * (M / (M_blocks))) * (K) + (threadIdx.z) * K_per_loop;
    // #pragma unroll
    // for(int mem_addr = thread_id + threadIdx.z * N_block_size * K_block_size; mem_addr < total_elements; mem_addr += N_block_size * K_block_size * M_block_size){
    //   cooperative_groups::memcpy_async(cooperative_groups::this_thread(), 
    //   &reinterpret_cast<int4*>(A_shared)[mem_addr], 
    //   reinterpret_cast<int4*>(&A[start + (mem_addr % (M / M_blocks)) * K]), 
    //   sizeof(int4));
    // }     
    // cooperative_groups::wait(block);
    // block.sync();

    #pragma unroll
    for (int k_0 = 0; k_0 < K_loop_end; ++k_0) {      
      B_local[0] = *(int*)(B + 
        (((n_0 * N_blocks + blockIdx.x) * M_block_size + threadIdx.z) * (N_per_loop) * K / 4) + 
        (k_0 * K_loop_num * K_per_loop * N_per_loop / 4) + src_offset);     

      start = (blockIdx.y * (M / (M_blocks))) * (K) + (k_0 + threadIdx.z) * K_per_loop;
      if((k_0 % M_block_size) == 0){
        #pragma unroll
        for(int mem_addr = thread_id + threadIdx.z * N_block_size * K_block_size; mem_addr < total_elements; mem_addr += N_block_size * K_block_size * M_block_size){
          cooperative_groups::memcpy_async(cooperative_groups::this_thread(), 
          &reinterpret_cast<int4*>(&A_shared)[mem_addr], 
          reinterpret_cast<int4*>(&A[start + (mem_addr % (M / M_blocks)) * K]), 
          sizeof(int4));
        } 
      }

      
      // if(k_0 < K_loop_end - 1){
      //   cooperative_groups::memcpy_async(cooperative_groups::this_thread(), 
      //   reinterpret_cast<int*>(&B_shared_buf[(threadIdx.z * buf_size * N_block_size * K_block_size + threadIdx.y * K_block_size + threadIdx.x + ((buf_idx + 1) % buf_size) * N_block_size * K_block_size)]), 
      //   reinterpret_cast<int*>(B + (((n_0 * N_blocks + blockIdx.x) * M_block_size + threadIdx.z) * (N_per_loop) * K / 4) + ((k_0 + 1) * K_loop_num * K_per_loop * N_per_loop / 4) + src_offset), 
      //   sizeof(int));
      // }
      
      // if((buf_idx) % M_block_size == 0){
      //   start = (blockIdx.y * (M / (M_blocks))) * (K) + (k_0 + M_block_size + threadIdx.z) * K_per_loop; 
      //   #pragma unroll
      //   for(int mem_addr = thread_id + threadIdx.z * N_block_size * K_block_size; mem_addr < total_elements; mem_addr += N_block_size * K_block_size * M_block_size){
      //     cooperative_groups::memcpy_async(cooperative_groups::this_thread(), 
      //     &reinterpret_cast<int4*>(&A_shared[((buf_idx + M_block_size) % (buf_size * M_block_size)) * (M / M_blocks) * K_per_loop])[mem_addr], 
      //     reinterpret_cast<int4*>(&A[start + (mem_addr % (M / M_blocks)) * K]), 
      //     sizeof(int4));
      //   } 
      // }
      // B_local[0] = B_shared_buf[(threadIdx.z * buf_size * N_block_size * K_block_size + threadIdx.y * K_block_size + threadIdx.x + ((buf_idx % buf_size) * N_block_size * K_block_size))];

      decode_i2s_to_i8s(B_local, B_decode_local, 16);
      *reinterpret_cast<int4*>(&B_decode[B_shared_start + dst_offset]) = *reinterpret_cast<int4*>(B_decode_local);      

      #pragma unroll
      for(int k_2_0 = 0; k_2_0 < K_loop_num; ++k_2_0){
        wmma::load_matrix_sync(b_frag, &B_decode[B_shared_start + k_2_0 * K_per_loop * N_per_loop], K_per_loop);
        if(k_2_0 == 0){
          cooperative_groups::wait(block);
        }
        # pragma unroll
        for(int m_0 = 0; m_0 < M_loop_end; ++m_0){
          wmma::load_matrix_sync(a_frag, &A_shared[m_0 * M_per_loop * K_per_loop + (k_0 % M_block_size) * (M/M_blocks) * K_per_loop], K_per_loop);
          // wmma::load_matrix_sync(a_frag, &A_shared[m_0 * M_per_loop * K_per_loop + buf_idx * (M / M_blocks) * K_per_loop], K_per_loop);

          wmma::mma_sync(c_frag[m_0], a_frag, b_frag, c_frag[m_0]);      
        }
      }
      // buf_idx = (buf_idx + 1) % (buf_size * M_block_size);
      // cooperative_groups::wait(block);
      block.sync();      
    }
    base_idx = write_mem_idx + (n_0 * N_blocks + blockIdx.x) * M_block_size  * N_per_loop;
    #pragma unroll
    for(int m_0 = 0; m_0 < M_loop_end; ++m_0){
      wmma::store_matrix_sync(&Dump_buff[dump_buff_offset], c_frag[m_0], N_per_loop, wmma::mem_row_major);

      #pragma unroll
      for(int idx = thread_id; idx < n_vals; idx = idx + N_block_size * K_block_size){
        __nv_bfloat16 out = ((__nv_bfloat16)(Dump_buff[dump_buff_offset + idx]) / (s_shared[m_0 * M_per_loop + (idx / N_per_loop)]));
        // __nv_bfloat16 out = ((__nv_bfloat16)(Dump_buff[dump_buff_offset + m_0 * M_per_loop * N_per_loop + idx ]) / (s_shared[m_0 * M_per_loop + (idx / N_per_loop)]));

        dtype_transform[base_idx + ((idx / N_per_loop) + m_0 * M_per_loop) * N + thread_id] = out;
      }
    }
  }
}
