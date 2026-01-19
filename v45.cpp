#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>

#ifndef CHANNELS_PER_BLOCK
#define CHANNELS_PER_BLOCK 32
#endif
#ifndef POS_TILE
#define POS_TILE 256 // 用于沿 N*outH*outW 做并行累加的线程数
#endif
#ifndef BINARY_TREE
#define BINARY_TREE 0
#endif

typedef __fp16 HALF_T;
using fp16xN  = __fp16 __attribute__((ext_vector_type(CHANNELS_PER_BLOCK)));
using floatxN = float __attribute__((ext_vector_type(CHANNELS_PER_BLOCK)));

struct WrwParam
{
    const HALF_T* in_nhwc;  // [N,H,W,C]
    const HALF_T* out_nhwc; // [N,outH,outW,C]
    float* dW;              // [C,R,S]（展平：C*R*S）
    int N, C, H, W;
    int outH, outW;
    int R, S;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilate_h, dilate_w;
};

// 将一维 pos 解码为 (n, oh, ow)
__device__ __forceinline__ void decode_pos(int pos, int outH, int outW, int& oh, int& ow)
{
    oh = pos / outW;
    ow = pos - oh * outW;
}

struct Pos3D
{
    int n, oh, ow;
};

__device__ __forceinline__ Pos3D decode_pos_struct(int pos, int outH, int outW)
{
    int hw = outH * outW;
    Pos3D r;
    r.n     = pos / hw;
    int rem = pos - r.n * hw;
    r.oh    = rem / outW;
    r.ow    = rem - r.oh * outW;
    return r;
}

__device__ float sum_fp32_cross_lane_stride16(const float val, const int bpermute_addr)
{
    float sum = val;
    sum += __hip_ds_swizzlef_N<0x401F>(sum);       // tid0:0+16 tid32: 32+48
    sum += __hip_ds_bpermutef(bpermute_addr, sum); // tid0:0+16+32+48
    return sum;
}

// ------------------------------------
__global__ void dw_wrw_nhwc_grid_crs_kernel(const WrwParam p)
{
    const int c_block_start = blockIdx.x * CHANNELS_PER_BLOCK;

    const int tx = threadIdx.x; // 沿 (N*outH*outW) 并行
    extern __shared__ HALF_T smem;

    // 位置空间总元素
    const int POS = p.outH * p.outW;

    // 每个线程累加自己对应通道（或通道对）的部分
    floatxN acc[9] = {0.0f};

    for(int n = 0; n < p.N; n++)
    {
        for(int idx = tx * CHANNELS_PER_BLOCK; idx < p.H * p.W * CHANNELS_PER_BLOCK;
            idx += blockDim.x * CHANNELS_PER_BLOCK)
        {
            const HALF_T* in_ptr = p.in_nhwc + n * p.H * p.W + idx * c_block_start;
            *reinterpret_cast<const fp16xN*>(smem) = *reinterpret_cast<const fp16xN*>(in_ptr);
        }

        __syncthreads();

        for(int pos = tx; pos < POS; pos += blockDim.x)
        {
            int oh, ow;
            decode_pos(pos, p.outH, p.outW, oh, ow);

            for(int r = 0; r < p.R; r++)
            {
                for(int s = 0; s < p.S, s++)
                {
                    const int ih = oh * p.stride_h - p.pad_h + r * p.dilate_h;
                    const int iw = ow * p.stride_w - p.pad_w + s * p.dilate_w;

                    // 越界则跳过 小于0也跳过
                    if((unsigned)ih >= (unsigned)p.H || (unsigned)iw >= (unsigned)p.W)
                        continue;

                    const size_t in_base  = ih * p.W + iw;
                    const size_t out_base = ((size_t)n * p.outH + oh) * p.outW + ow;

                    const HALF_T* in_ptr  = smem + in_base * c_block_start + c_block_start;
                    const HALF_T* out_ptr = p.out_nhwc + out_base * c_block_start + c_block_start;

                    fp16xN in_h  = *reinterpret_cast<const fp16xN*>(in_ptr);
                    fp16xN out_h = *reinterpret_cast<const fp16xN*>(out_ptr);

#pragma unroll
                    for(int ch = 0; ch < CHANNELS_PER_BLOCK; ch++)
                    {
                        acc[ch][r * p.R + s] = fmaf(in_h[ch], out_h[ch], acc[ch]);
                    }
                }
            }
        }
    }

    const int wave_size = 64;
    int wave_id         = tx >> 6;
    int lane_id         = __lane_id();
    int num_wave        = blockDim.x / wave_size;

    float* buf = reinterpret_cast<float*>(smem + p.H * p.W * CHANNELS_PER_BLOCK);

    int32_t bpermute_addr = 0;
    for(int32_t id = 32; id > 0; id = id >> 1)
    {
        bpermute_addr = (lane_id ^ id) << 2;
#pragma unroll
        for(int ch = 0; ch < CHANNELS_PER_BLOCK; ch++)
            acc[ch] += __hip_ds_bpermutef(bpermute_addr, acc[ch]);
    }

    // 每个wave值写入lds中
    if(lane_id == 0)
    {
        *reinterpret_cast<floatxN*>(&buf[wave_id * CHANNELS_PER_BLOCK]) = acc;
    }

    __syncthreads();

    // 最终结果由线程0写入全局内存
    if(threadIdx.x == 0)
    {
        floatxN sum   = {0.0f};
        floatxN other = {0.0f};
#pragma unroll
        for(int w = 0; w < 4; w++)
        {
            other = *reinterpret_cast<floatxN*>(&buf[w * CHANNELS_PER_BLOCK]);
#pragma unroll
            for(int ch = 0; ch < CHANNELS_PER_BLOCK; ch++)
            {
                sum[ch] += other[ch];
            }
        }

#pragma unroll
        for(int ch = 0; ch < CHANNELS_PER_BLOCK; ch++)
        {
            size_t idx = (size_t)(c_block_start + ch) * (p.R * p.S) + r * p.S + s;
            p.dW[idx]  = sum[ch];
        }
    }
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

size_t calc_smem_bytes(int H, int W)
{
#if BINARY_TREE
    size_t elems = (size_t)POS_TILE * CHANNELS_PER_BLOCK;
#else
    int num_wave       = POS_TILE / 64;
    size_t elems       = (size_t)num_wave * CHANNELS_PER_BLOCK;
    size_t input_elems = (size_t)CHANNELS_PER_BLOCK * H * W;
#endif
    return input_elems * sizeof(HALF_T) + elems * sizeof(float);
}

void launch_dw_wrw_nhwc_grid_crs(const WrwParam& param, hipStream_t stream)
{
    dim3 grid((param.C + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK, 1, 1);

    dim3 block(POS_TILE, 1, 1);

    size_t smem_bytes = calc_smem_bytes(param.H, param.W);

    hipLaunchKernelGGL(dw_wrw_nhwc_grid_crs_kernel, grid, block, smem_bytes, stream, param);
}

int main()
{
    hipSetDevice(7);
    int N = 220, C = 192, H = 112, W = 112;
    int R = 3, S = 3;
    int pad_h = 1, pad_w = 1;
    int stride_h = 2, stride_w = 2;
    int dilate_h = 1, dilate_w = 1;

    N = 220, C = 3840, H = 7, W = 7;
    R = 3, S = 3;
    pad_h = 1, pad_w = 1;
    stride_h = 1, stride_w = 1;
    dilate_h = 1, dilate_w = 1;

    // N = 220, C = 1344, H = 14, W = 14;
    // R = 5, S = 5;
    // pad_h = 2, pad_w = 2;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;

    // N = 220, C = 480, H = 28, W = 28;
    // R = 5, S = 5;
    // pad_h = 2, pad_w = 2;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;

    // N = 220, C = 192, H = 112, W = 112;
    // R = 3, S = 3;
    // pad_h = 1, pad_w = 1;
    // stride_h = 2, stride_w = 2;
    // dilate_h = 1, dilate_w = 1;

    int outH = (H + 2 * pad_h - dilate_h * (R - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilate_w * (S - 1) - 1) / stride_w + 1;

    std::cout << "out(H" << outH << " W" << outW << ")" << std::endl;

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
    launch_dw_wrw_nhwc_grid_crs(param, stream);
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
        launch_dw_wrw_nhwc_grid_crs(param, stream);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    std::cout << "Average kernel time: " << total_ms / iters << " ms\n";

    // 清理
    hipStreamDestroy(stream);
    hipFree(d_input_nhwc);
    hipFree(d_output_nhwc);
    hipFree(d_weight_grad);
    return 0;
}
