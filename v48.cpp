#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
// #include "v35_temp_define.hpp"

#ifndef TILE_C
#define TILE_C 64
#endif
#ifndef TILE_OH
#define TILE_OH 8
#endif
#ifndef TILE_OW
#define TILE_OW 8
#endif
#ifndef MAX_R
#define MAX_R 5
#endif
#ifndef MAX_S
#define MAX_S 5
#endif
#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef CHANNELS_PER_THREAD
#define CHANNELS_PER_THREAD 8
#endif

typedef __fp16 HALF_T;
using fp16xN  = __fp16 __attribute__((ext_vector_type(CHANNELS_PER_THREAD)));
using float4_ = float __attribute__((ext_vector_type(4)));
using float2_ = float __attribute__((ext_vector_type(2)));

// 卷积参数结构（保留 NHWC）
struct WrwParam
{
    HALF_T* __restrict__ in_nhwc;  // [N,H,W,C]
    HALF_T* __restrict__ out_nhwc; // [N,outH,outW,C]
    float* dW;                     // [C,R,S]（展平：C*R*S）
    int N, C, H, W;
    int outH, outW;
    int R, S;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilate_h, dilate_w;
};

// 将一维 pos 解码为 (n, oh, ow)
__device__ __forceinline__ void decode_pos(int pos, int outH, int outW, int& n, int& oh, int& ow)
{
    int hw = outH * outW;
    n      = pos / hw;
    int t  = pos - n * hw;
    oh     = t / outW;
    ow     = t - oh * outW;
}

__host__ __device__ __forceinline__ int tile_in_extent(int tile_o, int stride, int dilate, int RorS)
{
    // 覆盖 oh in [0..tile_o-1] 与 r in [0..R-1]:
    // ih spans: oh*stride + r*dilate  => size = (tile_o-1)*stride + (R-1)*dilate + 1
    return (tile_o - 1) * stride + (RorS - 1) * dilate + 1;
}
__device__ __forceinline__ void mfma_f32_16x16x16f16(float4_& C, float2_& A, float2_& B)
{
    // asm volatile("v_mfma_f32_16x16x16f16 %0, %1, %2, %0\n\t" : "+v"(C), "+v"(A), "+v"(B));
    __builtin_amdgcn_mfma_f32_16x16x16f16(A, B, C, 0, 0, 0);
}

__device__ __forceinline__ void mmac_fp32_16x16x16_f16(float4_& C, float2_& A, float2_& B)
{
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t" : "+v"(C), "+v"(A), "+v"(B));
}

__global__
__launch_bounds__(WARP_SIZE* WARPS_PER_BLOCK) void dw_wrw_nhwc_tensorcore_kernel(const WrwParam p)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int c_blk   = blockIdx.x * TILE_C;
    const int oh_blk  = blockIdx.y * TILE_OH;
    const int ow_blk  = blockIdx.z * TILE_OW;

    if(c_blk >= p.C || oh_blk >= p.outH || ow_blk >= p.outW)
        return;

    // 计算实际处理的尺寸
    const int c_end  = min(c_blk + TILE_C, p.C);
    const int oh_end = min(oh_blk + TILE_OH, p.outH);
    const int ow_end = min(ow_blk + TILE_OW, p.outW);

    // 计算输入贴片大小
    const int H_in_tile = tile_in_extent(TILE_OH, p.stride_h, p.dilate_h, p.R);
    const int W_in_tile = tile_in_extent(TILE_OW, p.stride_w, p.dilate_w, p.S);

    // 共享内存分配
    extern __shared__ HALF_T smem[];
    HALF_T* sm_out = smem;
    HALF_T* sm_in  = sm_out + TILE_C * TILE_OH * TILE_OW;

    float acc[4] = {0.0f};

    for(int n = 0; n < p.N; ++n)
    {
        for(int i = threadIdx.x; i < TILE_C * TILE_OH * TILE_OW; i += blockDim.x)
        {
            int c_idx  = i % TILE_C;
            int oh_idx = (i / TILE_C) % TILE_OW;
            int ow_idx = i / (TILE_C * TILE_OW);

            int c  = c_blk + c_idx;
            int oh = oh_blk + oh_idx;
            int ow = ow_blk + ow_idx;

            if(c < p.C && oh < p.outH && ow < p.outW)
            {
                size_t out_idx = ((n * p.outH + oh) * p.outW + ow) * p.C + c;
                sm_out[i]      = p.out_nhwc[out_idx];
            }
            else
            {
                sm_out[i] = 0.0f;
            }
        }

        for(int i = threadIdx.x; i < TILE_C * H_in_tile * W_in_tile; i += blockDim.x)
        {
            int c_idx  = i % TILE_C;
            int ih_idx = (i / TILE_C) % H_in_tile;
            int iw_idx = i / (TILE_C * H_in_tile);

            int c  = c_blk + c_idx;
            int ih = oh_blk * p.stride_h - p.pad_h + ih_idx;
            int iw = ow_blk * p.stride_w - p.pad_w + iw_idx;

            if(c < p.C && ih >= 0 && ih < p.H && iw >= 0 && iw < p.W)
            {
                size_t in_idx = ((n * p.H + ih) * p.W + iw) * p.C + c;
                sm_in[i]      = p.in_nhwc[in_idx];
            }
            else
            {
                sm_in[i] = 0.0f;
            }
        }

        __syncthreads();

        for(int r = 0; r < p.R; ++r)
        {
            for(int s = 0; s < p.S; ++s)
            {
                int ih_offset = r * p.dilate_h;
                int iw_offset = s * p.dilate_w;

                if(ih_offset < H_in_tile && iw_offset < W_in_tile)
                {
                    size_t out_offset = lane_id * TILE_C;
                    size_t in_offset  = (ih_offset * W_in_tile + iw_offset) * TILE_C + lane_id;

                    __fp16 a0 = sm_out[out_offset];
                    __fp16 a1 = sm_out[out_offset + 1];
                    __fp16 a2 = sm_out[out_offset + 2];
                    __fp16 a3 = sm_out[out_offset + 3];

                    __fp16 b0 = sm_in[in_offset];
                    __fp16 b1 = sm_in[in_offset + 1];
                    __fp16 b2 = sm_in[in_offset + 2];
                    __fp16 b3 = sm_in[in_offset + 3];

                    // mfma_f32_16x16x16f16(
                    //     *reinterpret_cast<float4_ *>(acc),
                    //     *reinterpret_cast<float2_ *>(&a0),
                    //     *reinterpret_cast<float2_ *>(&b0));
                    mmac_fp32_16x16x16_f16(*reinterpret_cast<float4_*>(acc),
                                           *reinterpret_cast<float2_*>(&a0),
                                           *reinterpret_cast<float2_*>(&b0));
                }
            }
        }

        __syncthreads();
    }

    for(int r = 0; r < p.R; ++r)
    {
        for(int s = 0; s < p.S; ++s)
        {
            int c = c_blk + lane_id;
            if(c < p.C)
            {
                size_t w_idx = (c * p.R + r) * p.S + s;
                atomicAdd(&p.dW[w_idx], acc[lane_id % 4]);
            }
        }
    }
}

// 计算共享内存大小
size_t calc_smem_bytes_tensorcore(const WrwParam& p)
{
    const int H_in_tile = tile_in_extent(TILE_OH, p.stride_h, p.dilate_h, p.R);
    const int W_in_tile = tile_in_extent(TILE_OW, p.stride_w, p.dilate_w, p.S);

    size_t out_tile_elems = (size_t)TILE_C * TILE_OH * TILE_OW;
    size_t in_tile_elems  = (size_t)TILE_C * H_in_tile * W_in_tile;

    return (out_tile_elems + in_tile_elems) * sizeof(HALF_T);
}

// 启动函数
void launch_dw_wrw_nhwc_tensorcore(const WrwParam& param, hipStream_t stream)
{
    dim3 grid(1,
              (param.C + TILE_C - 1) / TILE_C,
              (param.outH + TILE_M - 1) / TILE_M * (param.outW + TILE_N - 1) / TILE_N);

    dim3 block(WARP_SIZE * WARPS_PER_BLOCK, 1, 1);

    size_t smem_bytes = calc_smem_bytes_tensorcore(param);

    hipLaunchKernelGGL(dw_wrw_nhwc_tensorcore_kernel, grid, block, smem_bytes, stream, param);
}

// ---------------------------
// Host-side 实现（保持 NHWC，无需转置）
// ---------------------------

#define COLOR_RED "\033[1;31m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_BLUE "\033[1;34m"
#define COLOR_RESET "\033[0m"

float random_float(float min = -1.0f, float max = 1.0f)
{
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (max - min));
}

void depthwise_conv_wrw_cpu_nhwc(const HALF_T* input_nhwc,  // NHWC
                                 const HALF_T* output_nhwc, // NHWC
                                 float* weight_grad,        // [C,R,S]
                                 int N,
                                 int H,
                                 int W,
                                 int C,
                                 int R,
                                 int S,
                                 int pad_h,
                                 int pad_w,
                                 int stride_h,
                                 int stride_w,
                                 int dilate_h,
                                 int dilate_w)
{
    int outH = (H + 2 * pad_h - dilate_h * (R - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilate_w * (S - 1) - 1) / stride_w + 1;

    int weight_size = C * R * S;
    for(int i = 0; i < weight_size; i++)
        weight_grad[i] = 0.0f;

    for(int n = 0; n < N; n++)
    {
        for(int oh = 0; oh < outH; oh++)
            for(int ow = 0; ow < outW; ow++)
                for(int c = 0; c < C; c++)
                {
                    int out_idx    = ((n * outH + oh) * outW + ow) * C + c;
                    float out_grad = __half2float(output_nhwc[out_idx]);

                    for(int r = 0; r < R; r++)
                        for(int s = 0; s < S; s++)
                        {
                            int ih = oh * stride_h - pad_h + r * dilate_h;
                            int iw = ow * stride_w - pad_w + s * dilate_w;

                            if((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                            {
                                int in_idx   = ((n * H + ih) * W + iw) * C + c;
                                float in_val = __half2float(input_nhwc[in_idx]);

                                int weight_idx = c * R * S + r * S + s;
                                weight_grad[weight_idx] += in_val * out_grad;
                            }
                        }
                }
    }
}

// 计算计算强度
void calculate_arithmetic_intensity(const WrwParam& param,
                                    double& flops,
                                    double& bytes,
                                    double& intensity)
{
    // 计算操作数: N * outH * outW * C * R * S * 2 (乘加操作)
    flops =
        static_cast<double>(param.N) * param.outH * param.outW * param.C * param.R * param.S * 2.0;

    // 内存访问量: 输入 + 输出 + 权重梯度
    double input_bytes =
        static_cast<double>(param.N) * param.H * param.W * param.C * sizeof(HALF_T);
    double output_bytes =
        static_cast<double>(param.N) * param.outH * param.outW * param.C * sizeof(HALF_T);
    double weight_bytes = static_cast<double>(param.C) * param.R * param.S * sizeof(float);

    bytes = input_bytes + output_bytes + weight_bytes;

    // 计算强度: FLOPs / Bytes
    intensity = flops / bytes;
}

bool compare_arrays(const float* a, const float* b, int size, float tolerance = 4e-2f)
{
    for(int i = 0; i < size; i++)
    {
        if(std::abs(a[i] - b[i]) > tolerance)
        {
            std::cout << COLOR_RED << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                      << " (diff: " << std::abs(a[i] - b[i]) << ")" << COLOR_RESET << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    hipSetDevice(1);
    int N = 220, C = 32, H = 28, W = 28;
    int R = 3, S = 3;
    int pad_h = 0, pad_w = 0;
    int stride_h = 1, stride_w = 1;
    int dilate_h = 1, dilate_w = 1;

    // N = 220, C = 480, H = 28, W = 28;
    // R = 5, S = 5;
    // pad_h = 2, pad_w = 2;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;

    N = 220, C = 192, H = 112, W = 112;
    R = 5, S = 5;
    pad_h = 1, pad_w = 1;
    stride_h = 2, stride_w = 2;
    dilate_h = 1, dilate_w = 1;

    // N = 220, C = 3840, H = 7, W = 7;
    // R = 3, S = 3;
    // pad_h = 1, pad_w = 1;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;

    // N = 220, C = 192, H = 112, W = 112;
    // R = 3, S = 3;
    // pad_h = 1, pad_w = 1;
    // stride_h = 2, stride_w = 2;
    // dilate_h = 1, dilate_w = 1;

    int outH = (H + 2 * pad_h - dilate_h * (R - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilate_w * (S - 1) - 1) / stride_w + 1;

    std::cout << "out=> H:" << outH << "; W: " << outW << std::endl;

    // Host buffers
    std::vector<float> weight_grad_cpu(C * R * S);
    std::vector<HALF_T> input_nhwc(N * H * W * C);
    std::vector<HALF_T> output_nhwc(N * outH * outW * C);

    srand(42);
    for(size_t i = 0; i < input_nhwc.size(); ++i)
        input_nhwc[i] = HALF_T(random_float(-1.0f, 1.0f));
    for(size_t i = 0; i < output_nhwc.size(); ++i)
        output_nhwc[i] = HALF_T(random_float(-1.0f, 1.0f));

    // Device buffers
    size_t input_size  = input_nhwc.size() * sizeof(HALF_T);
    size_t output_size = output_nhwc.size() * sizeof(HALF_T);
    size_t weight_size = (size_t)C * R * S * sizeof(float);

    HALF_T *d_input_nhwc = nullptr, *d_output_nhwc = nullptr;
    float* d_weight_grad = nullptr;

    hipMalloc(&d_input_nhwc, input_size);
    hipMalloc(&d_output_nhwc, output_size);
    hipMalloc(&d_weight_grad, weight_size);

    hipMemcpy(d_input_nhwc, input_nhwc.data(), input_size, hipMemcpyHostToDevice);
    hipMemcpy(d_output_nhwc, output_nhwc.data(), output_size, hipMemcpyHostToDevice);
    hipMemset(d_weight_grad, 0, weight_size);

    hipStream_t stream;
    hipStreamCreate(&stream);

    // 参数
    WrwParam param{};
    param.in_nhwc  = d_input_nhwc;
    param.out_nhwc = d_output_nhwc;
    param.dW       = d_weight_grad;
    param.N        = N;
    param.C        = C;
    param.H        = H;
    param.W        = W;
    param.outH     = outH;
    param.outW     = outW;
    param.R        = R;
    param.S        = S;
    param.pad_h    = pad_h;
    param.pad_w    = pad_w;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.dilate_h = dilate_h;
    param.dilate_w = dilate_w;

    // 启动
    launch_dw_wrw_nhwc_tensorcore(param, stream);
    hipStreamSynchronize(stream);

    // 检错
    hipError_t error = hipGetLastError();
    if(error != hipSuccess)
    {
        std::cerr << "Kernel execution failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }

    // 取回结果
    std::vector<float> weight_grad_gpu(C * R * S);
    hipMemcpy(weight_grad_gpu.data(), d_weight_grad, weight_size, hipMemcpyDeviceToHost);

    // CPU 参考
    depthwise_conv_wrw_cpu_nhwc(input_nhwc.data(),
                                output_nhwc.data(),
                                weight_grad_cpu.data(),
                                N,
                                H,
                                W,
                                C,
                                R,
                                S,
                                pad_h,
                                pad_w,
                                stride_h,
                                stride_w,
                                dilate_h,
                                dilate_w);

    bool match =
        compare_arrays(weight_grad_cpu.data(), weight_grad_gpu.data(), (int)weight_grad_gpu.size());
    std::cout << (match ? COLOR_GREEN : COLOR_RED)
              << (match ? "CPU and GPU results match!" : "CPU and GPU results do not match!")
              << COLOR_RESET << std::endl;

    // 简单性能测试
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    const int iters = 100;
    float total_ms  = 0.f;

    for(int i = 0; i < iters; ++i)
    {
        hipMemset(d_weight_grad, 0, weight_size);
        hipEventRecord(start, stream);
        launch_dw_wrw_nhwc_tensorcore(param, stream);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    std::cout << "Average kernel time: " << total_ms / iters << " ms\n";

    size_t smem_bytes = calc_smem_bytes_tensorcore(param);
    std::cout << "smem : " << smem_bytes << " bytes" << std::endl;

    double num_of_flops, num_bytes_access, intensity;
    calculate_arithmetic_intensity(param, num_of_flops, num_bytes_access, intensity);

    std::cout << "intensity: " << intensity << std::endl;
    // size_t bandwidth = 1.2 TB/s;
    // size_t ops = 14.3 Tflops;
    // std::cout << "ops:byte: " <<

    // 清理
    hipStreamDestroy(stream);
    hipFree(d_input_nhwc);
    hipFree(d_output_nhwc);
    hipFree(d_weight_grad);
    return 0;
}
