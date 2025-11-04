# nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --shared bitnet_kernels_new.cu -lcuda -gencode=arch=compute_80,code=compute_80 -o libbitnet.so
nvcc -std=c++17 \
     -use_fast_math \
     -Xcompiler="-O3 -fPIC" \
     --shared bitnet_kernels_8_32_16.cu \
     -lcuda \
     -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_80,code=compute_80 \
     -o libbitnet.so


# nvcc -std=c++17 \
#      -use_fast_math \
#      -Xcompiler="-O3 -fPIC" \
#      --shared bitnet_kernels_16_16_16.cu \
#      -lcuda \
#      -gencode=arch=compute_80,code=sm_80 \
#      -gencode=arch=compute_80,code=compute_80 \
#      -o libbitnet.so
