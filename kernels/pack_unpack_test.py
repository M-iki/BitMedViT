import numpy as np

def pack_2bit_weights(matrix_2bit):
    """Pack a 16x32 (or NxK, K%4==0) int8 matrix with values in [-2,1] to 2-bit (packed into int8, 4 per byte) row-major."""
    N, K = matrix_2bit.shape
    # Convert signed values [-2,-1,0,1] to unsigned [0,1,2,3]
    matrix_u2 = (matrix_2bit + 2).astype(np.uint8)
    packed = np.zeros((N, K//4), dtype=np.uint8)
    for n in range(N):
        for k4 in range(K//4):
            b = 0
            for j in range(4):
                b |= (matrix_u2[n, k4*4 + j] & 0x3) << (2*j)
            packed[n, k4] = b
    return packed

def unpack_2bit_weights(packed, K):
    """Unpack (N, K//4) array to (N, K) array with signed int8 in [-2,1]"""
    N = packed.shape[0]
    matrix_2bit = np.zeros((N, K), dtype=np.int8)
    for n in range(N):
        for k4 in range(K//4):
            b = packed[n, k4]
            for j in range(4):
                val = ((b >> (2*j)) & 0x3)
                matrix_2bit[n, k4*4 + j] = np.int8(val - 2)
    return matrix_2bit

N, K = 16, 32

orig = np.random.randint(-2, 2, size=(N, K), dtype=np.int8)

packed = pack_2bit_weights(orig)

unpacked = unpack_2bit_weights(packed, K)

print("Original:\n", orig)
print("Packed shape:", packed.shape)
print("Unpacked:\n", unpacked)
print("Pack/Unpack identical?", np.all(orig == unpacked))

