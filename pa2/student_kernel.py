KERNEL_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 3},
]

@triton.jit
def matmul_add_relu_kernel_fp16(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Step 1: Tile: Assignment
    #
    # Each kernel instance is mapped to a tile in the output matrix C.
    # Compute the starting indices (m_start, n_start) for this tile.
    # -------------------------------------------------------------------------
    # TODO: Compute the tile indices using program_id(0) for M and program_id(1) for N.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # tl.device_print("pid_m: ", pid_m)
    # tl.device_print("pid_n: ", pid_n)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    # tl.device_print("BLOCK_M: ", BLOCK_M)
    # tl.device_print("BLOCK_N: ", BLOCK_N)
    # tl.device_print("m_start: ", m_start)
    # tl.device_print("n_start: ", n_start)
    # tl.device_print("a_ptr: ", a_ptr)
    # tl.device_print("b_ptr: ", b_ptr)
    # -------------------------------------------------------------------------
    # Step 2: Register Tiling
    # -------------------------------------------------------------------------
    # TODO: Initialize the accumulator "acc" with zeros (dtype: float16 or float32).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)



    # -------------------------------------------------------------------------
    # Step 3: Shared Memory Tiling & Cooperative Fetching.
    # Compute pointers to the sub-tiles of A and B that are needed to compute
    # the current C tile. The offsets here serve to load BLOCK_M x BLOCK_K
    # and BLOCK_K x BLOCK_N blocks from A and B respectively.
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    offs_m = m_start + tl.arange(0, BLOCK_M)[:, None]
    offs_n = n_start + tl.arange(0, BLOCK_N)[None, :]
    for k in range(0, K, BLOCK_K):
        offs_k_a = k + tl.arange(0, BLOCK_K)[None, :]
        offs_k_b = k + tl.arange(0, BLOCK_K)[:, None]

        a_ptrs = a_ptr + offs_m * stride_am + offs_k_a * stride_ak
        b_ptrs = b_ptr + offs_k_b * stride_bk + offs_n * stride_bn

        a_mask = (offs_m < M) & (offs_k_a < K)
        b_mask = (offs_k_b < K) & (offs_n < N)

        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a_tile, b_tile, acc)

    # -------------------------------------------------------------------------
    # Step 4: Add C and Apply ReLU to the accumulator
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    c_ptrs = c_ptr + offs_m * stride_cm + offs_n * stride_cn
    c_mask = (offs_m < M) & (offs_n < N)
    c_tile = tl.load(c_ptrs, mask=c_mask, other=0.0)
    acc += c_tile
    acc = tl.maximum(acc, 0.0)

    # -------------------------------------------------------------------------
    # Step 5: Write Cache / Epilogue Fusion: Write the computed tile to D.
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    d_ptrs = d_ptr + offs_m * stride_dm + offs_n * stride_dn
    tl.store(d_ptrs, acc, mask=c_mask)
