#include "im2col.cuh"

#define MAX_GRIDDIM_Z 65535

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) {
        return;
    }

    const int64_t iic = i / (KH_KW);
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;

    const int64_t  iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OH; iz+=MAX_GRIDDIM_Z) {
        const int64_t  in = iz / OH;
        const int64_t  ioh = iz - in * OH;

        const int64_t iiw = iow * s0 + ikw * d0 - p0;
        const int64_t iih = ioh * s1 + ikh * d1 - p1;

        const int64_t offset_dst =
            ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[offset_dst] = 0.0f;
        } else {
            const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

// Specialized im2col kernel for the common embedding-conv case:
// - stride = 1
// - dilation = 1
// - symmetric padding p0=p1=1
//
// This removes some per-element arithmetic and branches in the hot loop.
template <typename T>
static __global__ void im2col_kernel_p1s1d1(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) {
        return;
    }

    const int64_t iic = i / (KH_KW);
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;

    const int64_t iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
        const int64_t in  = iz / OH;
        const int64_t ioh = iz - in * OH;

        // stride=1, dilation=1, padding=1
        const int64_t iiw = iow + ikw - 1;
        const int64_t iih = ioh + ikh - 1;

        const int64_t offset_dst = ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        // Interior outputs dominate; skip bounds checks when the 3x3 footprint is fully inside.
        // Interior condition for pad=1, k in [0..2]: out_x in [1..OW-2], out_y in [1..OH-2].
        if (iow > 0 && iow + 1 < OW && ioh > 0 && ioh + 1 < OH) {
            const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
        } else {
            if ((uint64_t) iih >= (uint64_t) IH || (uint64_t) iiw >= (uint64_t) IW) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
            }
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

// Out-parallel im2col for pad=1, stride=1, dil=1.
// One block per output position; threads cover IC*KH*KW in a loop.
// This reduces the number of blocks by a factor of ceil(IC*KH*KW / 256).
template <typename T>
static __global__ void im2col_kernel_p1s1d1_outpar(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW,
        int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t KH_KW, int64_t IC_KH_KW) {
    const int64_t out_idx = (int64_t) blockIdx.x;

    const int64_t iow = out_idx % OW;
    const int64_t iz  = out_idx / OW;
    const int64_t in  = iz / OH;
    const int64_t ioh = iz - in * OH;

    for (int64_t i = (int64_t) threadIdx.x; i < IC_KH_KW; i += (int64_t) blockDim.x) {
        const int64_t iic = i / KH_KW;
        const int64_t rem = i - iic * KH_KW;
        const int64_t ikh = rem / KW;
        const int64_t ikw = rem - ikh * KW;

        const int64_t iiw = iow + ikw - 1;
        const int64_t iih = ioh + ikh - 1;

        const int64_t offset_dst = out_idx * IC_KH_KW + i;

        if ((uint64_t) iih >= (uint64_t) IH || (uint64_t) iiw >= (uint64_t) IW) {
            dst[offset_dst] = 0.0f;
        } else {
            const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
        }
    }

    GGML_UNUSED(KH);
}



// Common case: stride=2, dilation=1, padding=1 (e.g., ResNet downsampling 3x3 conv).
template <typename T>
static __global__ void im2col_kernel_p1s2d1(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) {
        return;
    }

    const int64_t iic = i / (KH_KW);
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;

    const int64_t iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
        const int64_t in  = iz / OH;
        const int64_t ioh = iz - in * OH;

        // stride=2, dilation=1, padding=1
        const int64_t iiw = iow * 2 + ikw - 1;
        const int64_t iih = ioh * 2 + ikh - 1;

        const int64_t offset_dst = ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        if (iow > 0 && iow + 1 < OW && ioh > 0 && ioh + 1 < OH) {
            const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
        } else {
            if ((uint64_t) iih >= (uint64_t) IH || (uint64_t) iiw >= (uint64_t) IW) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
            }
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

template <typename T>
static __global__ void im2col_kernel_p1s2d1_outpar(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW,
        int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t KH_KW, int64_t IC_KH_KW) {
    const int64_t out_idx = (int64_t) blockIdx.x;

    const int64_t iow = out_idx % OW;
    const int64_t iz  = out_idx / OW;
    const int64_t in  = iz / OH;
    const int64_t ioh = iz - in * OH;

    for (int64_t i = (int64_t) threadIdx.x; i < IC_KH_KW; i += (int64_t) blockDim.x) {
        const int64_t iic = i / KH_KW;
        const int64_t rem = i - iic * KH_KW;
        const int64_t ikh = rem / KW;
        const int64_t ikw = rem - ikh * KW;

        const int64_t iiw = iow * 2 + ikw - 1;
        const int64_t iih = ioh * 2 + ikh - 1;

        const int64_t offset_dst = out_idx * IC_KH_KW + i;

        if ((uint64_t) iih >= (uint64_t) IH || (uint64_t) iiw >= (uint64_t) IW) {
            dst[offset_dst] = 0.0f;
        } else {
            const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
        }
    }

    GGML_UNUSED(KH);
}


// Common case: 1x1 kernel, dilation=1, padding=0, stride=1.
template <typename T>
static __global__ void im2col_kernel_k1_p0s1d1(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC) {
        return;
    }

    const int64_t iic = i;
    const int64_t iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
        const int64_t in  = iz / OH;
        const int64_t ioh = iz - in * OH;

        const int64_t iiw = iow;
        const int64_t iih = ioh;

        const int64_t offset_dst = ((in * OH + ioh) * OW + iow) * IC + iic;
        const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
        dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
    }

    GGML_UNUSED(IH);
}

// Common case: 1x1 kernel, dilation=1, padding=0, stride=2.
template <typename T>
static __global__ void im2col_kernel_k1_p0s2d1(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC) {
        return;
    }

    const int64_t iic = i;
    const int64_t iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
        const int64_t in  = iz / OH;
        const int64_t ioh = iz - in * OH;

        const int64_t iiw = iow * 2;
        const int64_t iih = ioh * 2;

        const int64_t offset_dst = ((in * OH + ioh) * OW + iow) * IC + iic;
        const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
        dst[offset_dst] = __ldg(x + (offset_src + iih * IW + iiw));
    }

    GGML_UNUSED(IH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t num_blocks = (IC_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    const int64_t N_OH = N * OH;
    const int64_t KH_KW = KW*KH;
    dim3 block_nums(num_blocks, OW, MIN(N_OH, MAX_GRIDDIM_Z));
    const int threads = MIN(IC_KH_KW, CUDA_IM2COL_BLOCK_SIZE);

    // Fast-paths for common embedding conv2d cases.
    if (KW == 3 && KH == 3 && s0 == 1 && s1 == 1 && d0 == 1 && d1 == 1 && p0 == 1 && p1 == 1) {
        if (IC_KH_KW > CUDA_IM2COL_BLOCK_SIZE) {
            const int64_t out_total = OW * N_OH;
            const dim3 grid((unsigned int) out_total, 1, 1);
            const dim3 block(CUDA_IM2COL_BLOCK_SIZE, 1, 1);
            im2col_kernel_p1s1d1_outpar<<<grid, block, 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                                    IC_IH_IW, IH_IW, KH_KW, IC_KH_KW);
        } else {
            im2col_kernel_p1s1d1<<<block_nums, threads, 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                                     IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW);
        }
    } else if (KW == 3 && KH == 3 && s0 == 2 && s1 == 2 && d0 == 1 && d1 == 1 && p0 == 1 && p1 == 1) {
        if (IC_KH_KW > CUDA_IM2COL_BLOCK_SIZE) {
            const int64_t out_total = OW * N_OH;
            const dim3 grid((unsigned int) out_total, 1, 1);
            const dim3 block(CUDA_IM2COL_BLOCK_SIZE, 1, 1);
            im2col_kernel_p1s2d1_outpar<<<grid, block, 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                                    IC_IH_IW, IH_IW, KH_KW, IC_KH_KW);
        } else {
            im2col_kernel_p1s2d1<<<block_nums, threads, 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                                     IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW);
        }
    } else if (KW == 1 && KH == 1 && d0 == 1 && d1 == 1 && p0 == 0 && p1 == 0) {
        // For 1x1, IC_KH_KW == IC.
        if (s0 == 1 && s1 == 1) {
            im2col_kernel_k1_p0s1d1<<<block_nums, threads, 0, stream>>>(x, dst, IC, IW, IH, OH, OW,
                                                                        IC_IH_IW, IH_IW, N_OH);
        } else if (s0 == 2 && s1 == 2) {
            im2col_kernel_k1_p0s2d1<<<block_nums, threads, 0, stream>>>(x, dst, IC, IW, IH, OH, OW,
                                                                        IC_IH_IW, IH_IW, N_OH);
        } else {
            im2col_kernel<<<block_nums, threads, 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                              IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                                              s0, s1, p0, p1, d0, d1);
        }
    } else {
        im2col_kernel<<<block_nums, threads, 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                          IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                                          s0, s1, p0, p1, d0, d1);
    }
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t N        = src1->ne[is_2D ? 3 : 2];
    const int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static  __global__ void im2col_3d_kernel(
        const float * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KD_KH_KW) {
        return;
    }
    GGML_UNUSED(N); GGML_UNUSED(OC); GGML_UNUSED(OH_OW); GGML_UNUSED(OD); GGML_UNUSED(OW); GGML_UNUSED(KD); GGML_UNUSED(KH);
    GGML_UNUSED(ID_IH_IW); GGML_UNUSED(IH_IW); GGML_UNUSED(IC_ID_IH_IW); GGML_UNUSED(OW_KD_KH_KW);

    const int64_t iic = i / KD_KH_KW;
    const int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    const int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    const int64_t ikw = i % KW;

    const int64_t  iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OD_OH; iz+=MAX_GRIDDIM_Z) {
        const int64_t in  = iz / OD_OH;
        const int64_t iod = (iz - in*OD_OH) / OH;
        const int64_t ioh = iz % OH;

        const int64_t iiw = iow * s0 + ikw * d0 - p0;
        const int64_t iih = ioh * s1 + ikh * d1 - p1;
        const int64_t iid = iod * s2 + ikd * d2 - p2;

        const int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

        if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
            dst[offset_dst] = 0.0f;
        } else {
            const int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
            dst[offset_dst] = src[offset_src];
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static void im2col_3d_cuda(const float * src, T* dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {
    const int64_t OH_OW = OH*OW;
    const int64_t KD_KH_KW = KD*KH*KW;
    const int64_t ID_IH_IW = ID*IH*IW;
    const int64_t KH_KW = KH*KW;
    const int64_t IH_IW = IH*IW;
    const int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    const int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    const int64_t N_OD_OH = N*OD*OH;
    const int64_t OD_OH = OD*OH;
    const int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    const int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    const int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    const int64_t num_blocks = (IC_KD_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, OW, MIN(N_OD_OH, MAX_GRIDDIM_Z));
    im2col_3d_kernel<<<block_nums, MIN(IC_KD_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                                                                                           OH_OW, KD_KH_KW, ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW,
                                                                                           IC_KD_KH_KW, OW_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW,
                                                                                           OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                                                                           stride_q, stride_z, stride_y, stride_x,
                                                                                           s0, s1, s2, p0, p1, p2, d0, d1, d2);
}

static void im2col_3d_cuda_f16(const float * src, half * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                         stride_q, stride_z, stride_y, stride_x,
                         s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

static void im2col_3d_cuda_f32(const float * src, float * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<float>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const size_t  es       = ggml_element_size(src1);
    const int64_t stride_x = src1->nb[0] / es;
    const int64_t stride_y = src1->nb[1] / es;
    const int64_t stride_z = src1->nb[2] / es;
    const int64_t stride_q = src1->nb[3] / es;

    if(dst->type == GGML_TYPE_F16) {
        im2col_3d_cuda_f16(src1_d, (half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } else {
        im2col_3d_cuda_f32(src1_d, (float *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
