#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>

#define PERFORM
#define BATCH_SINGLE_BLOCK 16
#define THREADS_BLOCK 256
#define ELEMENTS_PER_THREAD 8

using floatxN = float __attribute__((ext_vector_type(ELEMENTS_PER_THREAD)));

struct DepthWiseParam
{
    void* __restrict__ weightPtr;
    void* __restrict__ inPtr;
    void* __restrict__ outPtr;
    void* __restrict__ workspace;
    int n;
    int hi;
    int wi;
    int ho;
    int wo;
    int sy;
    int sx;
    int dy;
    int dx;
    int py;
    int px;
    int fy;
    int fx;
    int group;
};

__device__ float sum_fp32_cross_lane_stride16(const float val, const int bpermute_addr)
{
    float sum = val;
    sum += __hip_ds_swizzlef_N<0x401F>(sum);       // tid0:0+16 tid32: 32+48
    sum += __hip_ds_bpermutef(bpermute_addr, sum); // tid0:0+16+32+48
    return sum;
}

__global__ void depthwise_fwd(DepthWiseParam param)
{
    float* p_in  = (float*)param.inPtr;
    float* p_wei = (float*)param.weightPtr;
    float* p_out = (float*)param.outPtr;

    int hi    = param.hi;
    int wi    = param.wi;
    int n     = param.n;
    int ho    = param.ho;
    int wo    = param.wo;
    int fy    = param.fy;
    int fx    = param.fx;
    int group = param.group;

    const int threads_per_group = (fx + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;

    int t_col = threadIdx.x % threads_per_group;
    int t_row = threadIdx.x / threads_per_group;

    const int groups_single_block = THREADS_BLOCK / threads_per_group;

    int group_offset = blockIdx.x * groups_single_block;
    int batch_offset = blockIdx.y * BATCH_SINGLE_BLOCK;

    int current_group = group_offset + t_row;

    bool valid_group = (current_group < group) && (t_row < groups_single_block);
    bool valid_batch = (batch_offset < n);
    bool valid       = valid_group && valid_batch;

    size_t input_offset = (size_t)batch_offset * group * hi * wi + (size_t)current_group * hi * wi;
    size_t weight_offset = (size_t)current_group * fy * fx;
    size_t output_offset = (size_t)batch_offset * group * ho * wo + (size_t)current_group * ho * wo;

    extern __shared__ float smem[];
    float* smem_weight = smem;
    float* smem_sum    = smem_weight + (groups_single_block * fy * fx);

    floatxN in_vecx  = {0.0f};
    floatxN wei_vecx = {0.0f};

    int vec           = t_col * ELEMENTS_PER_THREAD;
    int elems_aligned = wi & ~(ELEMENTS_PER_THREAD - 1);

    if(valid && (vec + ELEMENTS_PER_THREAD - 1) < elems_aligned)
    {
        in_vecx  = *reinterpret_cast<const floatxN*>(p_in + input_offset + vec);
        wei_vecx = *reinterpret_cast<const floatxN*>(p_wei + weight_offset + vec);
    }
    else
    {
#pragma unroll
        for(int e = 0; e < ELEMENTS_PER_THREAD; ++e)
        {
            int idx = vec + e;
            if(valid && idx < wi)
            {
                in_vecx[e]  = p_in[input_offset + idx];
                wei_vecx[e] = p_wei[weight_offset + idx];
            }
        }
    }

    float partial_sum = 0.0f;
#pragma unroll
    for(int e = 0; e < ELEMENTS_PER_THREAD; ++e)
        partial_sum += in_vecx[e] * wei_vecx[e];

    int smem_weight_idx = t_row * fx;

#pragma unroll
    for(int e = 0; e < ELEMENTS_PER_THREAD; ++e)
    {
        int idx = vec + e;
        float w = 0.0f;
        if(valid && idx < fx)
            w = wei_vecx[e];
        smem_weight[smem_weight_idx + idx] = w;
    }

    __syncthreads();

    int sum_idx = t_row * threads_per_group + t_col;

    for(int b = 0; b < BATCH_SINGLE_BLOCK && (batch_offset + b) < n;)
    {
        smem_sum[sum_idx] = partial_sum;
        __syncthreads();

        if(t_col == 0 && valid_group)
        {
            float channel_sum = 0.0f;
#pragma unroll
            for(int i = 0; i < threads_per_group; ++i)
                channel_sum += smem_sum[t_row * threads_per_group + i];

            p_out[output_offset + b * group * ho * wo] = channel_sum;
        }

        ++b;
        if((batch_offset + b) >= n)
            break;

        in_vecx = {0.0f};

        size_t next_input_offset = input_offset + (size_t)b * group * hi * wi;

        if(valid_group && (vec + ELEMENTS_PER_THREAD - 1) < elems_aligned)
        {
            in_vecx = *reinterpret_cast<const floatxN*>(p_in + next_input_offset + vec);
        }
        else
        {
#pragma unroll
            for(int e = 0; e < ELEMENTS_PER_THREAD; ++e)
            {
                int idx = vec + e;
                if(valid_group && idx < wi)
                    in_vecx[e] = p_in[next_input_offset + idx];
            }
        }

        float next_partial_sum = 0.0f;
#pragma unroll
        for(int e = 0; e < ELEMENTS_PER_THREAD; ++e)
            next_partial_sum += smem_weight[smem_weight_idx + vec + e] * in_vecx[e];

        partial_sum = next_partial_sum;
    }
}

// ---------------------------------------
/// Host-side
// ---------------------------------------

void depthwise_cpu_fwd(float* p_in,
                       float* p_wei,
                       float* p_out,
                       int n,
                       int c,
                       int hi,
                       int wi,
                       int ho,
                       int wo,
                       int fy,
                       int fx,
                       int sy,
                       int sx,
                       int py,
                       int px,
                       int dy,
                       int dx,
                       int group)
{
    for(int in_idx = 0; in_idx < n; ++in_idx)
    {
        for(int ic = 0; ic < c; ++ic)
        {
            for(int oh = 0; oh < ho; ++oh)
            {
                for(int ow = 0; ow < wo; ++ow)
                {
                    float sum = 0.0f;

                    for(int fh = 0; fh < fy; ++fh)
                    {
                        for(int fw = 0; fw < fx; ++fw)
                        {
                            int ih = oh * sy - py + fh * dy;
                            int iw = ow * sx - px + fw * dx;

                            if(ih >= 0 && ih < hi && iw >= 0 && iw < wi)
                            {
                                size_t input_idx =
                                    in_idx * c * hi * wi + ic * hi * wi + ih * wi + iw;
                                size_t weight_idx = ic * fy * fx + fh * fx + fw;
                                sum += p_in[input_idx] * p_wei[weight_idx];
                            }
                        }
                    }

                    size_t output_idx = in_idx * c * ho * wo + ic * ho * wo + oh * wo + ow;
                    p_out[output_idx] = sum;
                }
            }
        }
    }
}

#define COLOR_RED "\033[1;31m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_BLUE "\033[1;34m"
#define COLOR_RESET "\033[0m"

float random_float(float min = -1.0f, float max = 1.0f)
{
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (max - min));
}

bool compare_arrays(const float* a, const float* b, int size, float tolerance = 1e-3f)
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

template <typename T>
void print_matrix(T* data, size_t len, size_t start, size_t end)
{
    for(size_t i = start; i < std::min(len, end); ++i)
    {
        printf("%6.5f, ", data[i]);
        if(i % 32 == 31)
        {
            printf("\n");
        }
    }

    return;
}

int main(int argc, char** argv)
{
    int device_id = 1;
    hipSetDevice(device_id);

    int N = 1, C = 301500, H = 1, W = 53;
    int R = 1, S = 53;
    int pad_h = 0, pad_w = 0;
    int stride_h = 1, stride_w = 1;
    int dilate_h = 1, dilate_w = 1;
    int K = C;

    N = 201, C = 79500, H = 1, W = 53;
    R = 1, S = 53;
    pad_h = 0, pad_w = 0;
    stride_h = 1, stride_w = 1;
    dilate_h = 1, dilate_w = 1;
    K = C;

    // N = 1, C = 37, H = 1, W = 53;
    // R = 1, S = 53;
    // pad_h = 0, pad_w = 0;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;
    // K = C;

    // N = 1, C = 60, H = 256, W = 256;
    // R = 1, S = 1;
    // pad_h = 0, pad_w = 0;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;
    // K = C;

    // N = 478, C = 96, H = 1, W = 200;
    // R = 1, S = 200;
    // pad_h = 0, pad_w = 0;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;
    // K = C;

    int outH = (H + 2 * pad_h - dilate_h * (R - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilate_w * (S - 1) - 1) / stride_w + 1;

    // --------------------------------------------
    /// 入参限制条件
    /// outH == outW == 1
    /// H * W <= 64; R * S <= 64; outH * outW == 1
    // --------------------------------------------

    // if (H * W > 64 || R * S > 64 || outH * outW != 1 || H != R || W != S || H != 1 || R != 1)
    // {
    //     std::cout << "入参错误!!!检查入参限制条件,该kernel特定优化" << std::endl;
    //     return 1;
    // }

    std::cout << "output size: " << outH << ", " << outW << std::endl;
    std::cout << "\n";

    size_t input_size  = N * C * H * W;
    size_t weight_size = K * R * S;
    size_t output_size = N * K * outH * outW;

    printf("%zu\n", input_size);

    std::vector<float> input_host(input_size);
    std::vector<float> weight_host(weight_size);
    std::vector<float> output_host(output_size, 0.0f);

    for(size_t i = 0; i < input_host.size(); ++i)
    {
        input_host[i] = random_float(-1.0f, 1.0f);
    }

    for(size_t i = 0; i < weight_host.size(); ++i)
    {
        weight_host[i] = random_float(-1.0f, 1.0f);
    }

    float *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
    hipMalloc(&d_input, input_size * sizeof(float));
    hipMalloc(&d_weight, weight_size * sizeof(float));
    hipMalloc(&d_output, output_size * sizeof(float));

    hipMemcpy(d_input, input_host.data(), input_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_weight, weight_host.data(), weight_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, output_size * sizeof(float));

    DepthWiseParam param;
    param.weightPtr = d_weight;
    param.inPtr     = d_input;
    param.outPtr    = d_output;
    param.workspace = nullptr;
    param.hi        = H;
    param.wi        = W;
    param.n         = N;
    param.ho        = outH;
    param.wo        = outW;
    param.sy        = stride_h;
    param.sx        = stride_w;
    param.dy        = dilate_h;
    param.dx        = dilate_w;
    param.py        = pad_h;
    param.px        = pad_w;
    param.fy        = R;
    param.fx        = S;
    param.group     = C; // depthwise: group = channels

    const int threads_in_group    = (S + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    const int threads_col         = threads_in_group;
    const int threads_row         = THREADS_BLOCK / threads_in_group;
    const int weight_single_block = threads_row * R * S;
    const int reduce_size         = threads_col * threads_row;
    size_t dynamic_lds_size       = sizeof(float) * (weight_single_block + reduce_size);

    const int groups_single_block = threads_row;
    int grid_dim_x                = (C + groups_single_block - 1) / groups_single_block;
    int grid_dim_y                = (N + BATCH_SINGLE_BLOCK - 1) / BATCH_SINGLE_BLOCK;

    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(THREADS_BLOCK, 1, 1);

    std::cout << "blockDim(" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")"
              << "\n"
              << "gridDim(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")"
              << std::endl;

    std::cout << "\ndynamic_lds_size = " << dynamic_lds_size << std::endl;

    hipLaunchKernelGGL(depthwise_fwd, grid_dim, block_dim, dynamic_lds_size, 0, param);

    std::vector<float> outputGPUHostTensor(output_size);
    hipMemcpy(
        outputGPUHostTensor.data(), d_output, output_size * sizeof(float), hipMemcpyDeviceToHost);

    depthwise_cpu_fwd(input_host.data(),
                      weight_host.data(),
                      output_host.data(),
                      N,
                      C,
                      H,
                      W,
                      outH,
                      outW,
                      R,
                      S,
                      stride_h,
                      stride_w,
                      pad_h,
                      pad_w,
                      dilate_h,
                      dilate_w,
                      C);

    bool match = compare_arrays(outputGPUHostTensor.data(), output_host.data(), (int)output_size);

    if(match)
    {
        std::cout << "\n"
                  << COLOR_GREEN << " SUCCESS: All results match between CPU and GPU!"
                  << COLOR_RESET << std::endl;
    }
    else
    {
        std::cout << "\n"
                  << COLOR_RED << " FAILURE: Results DO NOT match between CPU and GPU!"
                  << COLOR_RESET << std::endl;
    }

    std::cout << "\n------------------------------ HOST output RES ----------------------------"
              << std::endl;
    print_matrix<float>(&output_host[0], output_host.size(), 0, 128);

    std::cout << "\n------------------------------ DEVICE output RES ----------------------------"
              << std::endl;
    print_matrix<float>(&outputGPUHostTensor[0], outputGPUHostTensor.size(), 0, 128);

    std::cout << "\n";

#ifdef PERFORM
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float cur_time   = 0.0f;
    float total_time = 0.0f;

    for(int iter = 0; iter < 100; iter++)
    {
        hipEventRecord(start, 0);
        hipLaunchKernelGGL(depthwise_fwd, grid_dim, block_dim, dynamic_lds_size, 0, param);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&cur_time, start, stop);
        total_time += cur_time;
    }

    float avg_time_ms = total_time / 100;
    printf("kernel time(ms): " COLOR_BLUE "%7.5f" COLOR_RESET "\n", avg_time_ms);

    float avg_time_s = avg_time_ms * 1e-3;

    ///
    float theoretical_bandwidth_gbs = 1.5e3;                           // 1.5TB/s = 1500 GB/s
    float theoretical_bandwidth_Bps = theoretical_bandwidth_gbs * 1e9; // Bytes/s

    size_t total_io_bytes  = sizeof(float) * (input_size + weight_size + output_size);
    size_t actual_io_bytes = sizeof(float) * (input_size + grid_dim_y * weight_size + output_size);

    float actual_bandwidth_gbs  = (actual_io_bytes * 1e-9) / avg_time_s;
    float bandwidth_utilization = actual_bandwidth_gbs / theoretical_bandwidth_gbs;

    std::cout << "\ntheoretical_bandwidth_gbs = " << theoretical_bandwidth_gbs << " GB/s"
              << "\nactual_io_gb = " << actual_io_bytes * 1e-9 << " GB"
              << "\nactual_bandwidth_gbs = " << actual_bandwidth_gbs << " GB/s"
              << "\nbandwidth_utilization = " << bandwidth_utilization * 100 << " %" << std::endl;

    // 计算访存比 算术强度
    size_t total_flops         = input_size * 2; // (out += iw * fx)2op * g * batch
    float arithmetic_intensity = (float)total_flops / (float)actual_io_bytes; // 算术强度

    ///
    float peak_flops_TFLOPS = 7.3f; // flops  7.3TFlops
    float peak_flops_FLOPS  = peak_flops_TFLOPS * 1e9;
    float op_byte_ratio     = peak_flops_FLOPS / theoretical_bandwidth_Bps; // 计算访存比

    std::cout << "\narithmetic_intensity = " << arithmetic_intensity << " flops/Bytes (ops/bytes)"
              << "\nop_byte_ratio = " << op_byte_ratio << " flops/Bytes (BW_math/BW_mem)"
              << std::endl;

    if(arithmetic_intensity > op_byte_ratio)
    {
        std::cout << "math limit!" << std::endl;
    }
    else
    {
        std::cout << "memory limit!" << std::endl;
    }

#endif

    hipFree(d_input);
    hipFree(d_weight);
    hipFree(d_output);

    return 0;
}
