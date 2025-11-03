import torch
from torch.utils import benchmark
from torch import nn

from pack_weight import convert_weight_int8_to_int2
from torch.profiler import profile, record_function, ProfilerActivity
import ctypes
import numpy as np
# set all seed
torch.manual_seed(42)
np.random.seed(42)

bitnet_lib = ctypes.CDLL('libbitnet.so')

def bitnet_int8xint2_linear(input0, input1, s, ws):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()
    
    # M_stride = 8
    # K_per_loop = 16
    # input0 = input0.reshape(int(M / M_stride), M_stride, int(K / (K_per_loop)), K_per_loop)
    # input0 = input0.permute(0, 2, 1, 3).contiguous()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4


    ret = torch.empty((1, M, N), dtype=torch.bfloat16, device=input0.device)
    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])

    return ret

if __name__ == '__main__':
    test_list = [
        #(2560,  2560), 
        #(3840,  2560), 
        #(13824, 2560),
        #(2560,  6912) ,
        #(3200, 3200), 
        #(4800, 3200), 
        #(3200, 10240),
        #(20480, 3200),
        (512, 2048),
        (2048, 512),
    ]
    M = 1024
    for N,K in test_list:
        weight = torch.randint(-1, 2, (N, K), dtype=torch.int8, device='cuda')
        # weight = torch.ones((N, K), dtype=torch.int8, device='cuda')

        weight_compressed = convert_weight_int8_to_int2(weight).to('cuda')

        orig_shape = weight_compressed.shape
        weight_compressed = weight_compressed.reshape([N // 16 // 2, 2, K // 16, 2, 8, 4])
        weight_compressed = weight_compressed.permute([0, 2, 3, 1, 4, 5])
        weight_compressed = weight_compressed.reshape([N // 16 // 2, K // 16, 4, 8, 4]).contiguous()

        n_cols = weight_compressed[:, :, [0, 1]]
        n_cols = n_cols.reshape([N // 16 // 2, K // 16 // 2, 4, 8, 4])

        k_cols = weight_compressed[:, :, [2, 3]]
        k_cols = k_cols.reshape([N // 16 // 2, K // 16 // 2, 4, 8, 4])

        weight_compressed = torch.concat([n_cols, k_cols], dim = 2).contiguous()
        weight_compressed = weight_compressed.view(N, K // 4).contiguous()


        for i in range(1):
            input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
            s = 0.1 + 0.9 * torch.rand((1, M, 1), dtype=torch.bfloat16, device='cuda')
            # s = torch.ones(1, M, 1, dtype=torch.bfloat16, device='cuda')
            # ws = torch.ones(1, dtype=torch.bfloat16, device='cuda')
            ws = 0.1 + 0.9 * torch.rand(1, dtype=torch.bfloat16, device='cuda')

            input0 = torch.randint(-128, 127, (1, M, K), dtype=torch.int8, device='cuda')
            # input0 = torch.ones((1, M, K), dtype=torch.int8, device='cuda')

            input0_bf16 = input0.to(torch.bfloat16)
            input_np = input0.cpu().to(torch.int32).numpy()
            weight_np = weight.cpu().to(torch.int32).T.numpy()
            out_np = np.matmul(input_np,weight_np)
            out_np = torch.tensor(out_np).cuda().to(torch.bfloat16) / (ws * s)
            # input_np = input0.cpu().to(torch.int32).numpy()
            # weight_np = weight.cpu().to(torch.int32).T.numpy()
            # out_np = np.matmul(input_np, weight_np)
            # out_np = torch.tensor(out_np).cuda().to(torch.bfloat16)

            out = bitnet_int8xint2_linear(input0, weight_compressed, s, ws)
            total = 0
            for row in range(1):#len(out[0])):
                total += torch.allclose(out[0][row, 10:], out_np[0][row, 10:], atol=1e-1)
                # print(out[0][row])
                # if(not torch.allclose(out[0][row][:-2], out_np[0][row][:-2], atol=1e-1)):
                #     print(out[0][row])
            # print(total)
            # print(f'custom == np {torch.all(out==out_np)}')
            print(f'custom == np {torch.allclose(out, out_np, atol=1e-1)}')


        #input0 = torch.randint(-128,127,(1, K),dtype=torch.int8, device='cuda')
        input0 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
        input0_fp16 = input0.to(torch.float16)
        input0_bf16 = input0.to(torch.bfloat16)
        weight_fp16 = weight.to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        # input_np = input0.cpu().to(torch.int32).numpy()
        # weight_np = weight.cpu().to(torch.int32).T.numpy()

        ret = torch.empty((M, N), dtype=torch.bfloat16, device=input0.device)
        #ret = torch.empty((1,N), dtype=torch.bfloat16, device=input0.device)
        #print(out, out_np)
        # print(out.mean(), out_np.mean())
        s = torch.ones((1, M, 1), dtype=torch.bfloat16, device='cuda')
        ws = torch.ones(1, dtype=torch.bfloat16, device='cuda')
        t0 = benchmark.Timer(
            stmt="bitnet_int8xint2_linear(input0, weight_compressed, s, ws)",
            setup="from __main__ import input0, weight_compressed, s, ws, bitnet_int8xint2_linear",
            num_threads=1,
        )

        t1 = benchmark.Timer(
            stmt="torch.nn.functional.linear(input0_bf16,weight_bf16) / s / ws",
            setup="from __main__ import input0_bf16, weight_bf16, s, ws",
            num_threads=1,
        )

        # t1 = benchmark.Timer(
        #     stmt="torch.matmul(input0_bf16,weight_fp16.T) / s / ws",
        #     setup="from __main__ import input0_bf16, weight_fp16, s, ws",
        #     num_threads=1,
        # )

        time0 = t0.timeit(50)
        time1 = t1.timeit(50)

        print(f'Shape{N,K}, W2A8: {time0.mean * 1e6:.2f}us, torch BF16: {time1.mean * 1e6:.2f}us')
        # activities = [ ProfilerActivity.CUDA, 
        #             #   ProfilerActivity.CPU
        #               ]
        # sort_by_keyword = 'cuda' + "_time_total"
        # with profile(activities=activities, record_shapes=True) as prof:
        #     with record_function("model_inference1"):
        #         for _ in range(10):
        #             bitnet_int8xint2_linear(input0, weight_compressed, s, ws, ret)
        #             torch.matmul(input0_fp16,weight_fp16)
        #             torch.matmul(input0_bf16,weight_bf16)

        # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=15))
        
