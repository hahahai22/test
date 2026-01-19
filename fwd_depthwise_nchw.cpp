#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>

#define VERIFY
#define ELEMENTS_PER_THREAD 8
using floatxN = float __attribute__((ext_vector_type(ELEMENTS_PER_THREAD)));

#define BATCHS_SINGLE_BLOCK 1
#define TILE_OH 8
#define TILE_OW 8

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
    int split_h;
    int split_w;
};

__device__ __forceinline__ float sum_fp32_cross_lane_stride16(const float val,
                                                              const int bpermute_addr)
{
    float sum = val;
    sum += __hip_ds_swizzlef_N<0x401F>(sum);       // tid0:0+16 tid32: 32+48
    sum += __hip_ds_bpermutef(bpermute_addr, sum); // tid0:0+16+32+48
    return sum;
}

__host__ __device__ __forceinline__ int tile_in_extent(int tile_o, int stride, int dilate, int RorS)
{
    // 覆盖 oh in [0..tile_o-1] 与 r in [0..R-1]:
    // ih spans: oh*stride + r*dilate  => size = (tile_o-1)*stride + (R-1)*dilate + 1
    return (tile_o - 1) * stride + (RorS - 1) * dilate + 1;
}

__global__ void depthwise_fwd(DepthWiseParam param)
{
    //------------------------------------
    float* p_in  = (float*)param.inPtr;
    float* p_wei = (float*)param.weightPtr;
    float* p_out = (float*)param.outPtr;
    int hi       = param.hi;
    int wi       = param.wi;
    int n        = param.n;
    int ho       = param.ho;
    int wo       = param.wo;
    int sy       = param.sy;
    int sx       = param.sx;
    int dy       = param.dy;
    int dx       = param.dx;
    int py       = param.py;
    int px       = param.px;
    int fy       = param.fy;
    int fx       = param.fx;
    int group    = param.group;
    int split_h  = param.split_h;
    int split_w  = param.split_w;
    // -------------------------------------
    /// 此kernel针对c=k=group=1特殊优化
    // -------------------------------------
    extern __shared__ float smem[];
    float* weight_smem  = smem;
    float* in_tile_smem = smem + (fy * fx);

    int wo_tile_idx = blockIdx.x;
    int ho_tile_idx = blockIdx.y;

    /// TODO:先单block处理一个batch，后续单block处理多个batch
    int batch_idx = blockIdx.z;

    const int H_in_tile = tile_in_extent(TILE_OH, sy, dy, fy);
    const int W_in_tile = tile_in_extent(TILE_OW, sx, dx, fx);

    /// 数据复用: 在batch和计算output不同spatial_dim上点时候进行复用
    // 注意！有余数！！！
    for(int vec = threadIdx.x * ELEMENTS_PER_THREAD; vec < fy * fx;
        vec += (blockDim.x * ELEMENTS_PER_THREAD))
    {
        if(vec + ELEMENTS_PER_THREAD <= fy * fx)
        {
            floatxN weightxN = *reinterpret_cast<floatxN*>(p_wei + vec);
            *reinterpret_cast<floatxN*>(weight_smem + vec) = weightxN;
        }
        else // tail  // 或者这里可以对smem初始化为0，补齐为整数倍，减少不必要的判断
        {
            for(int i = 0; i < ELEMENTS_PER_THREAD && (vec + i) < fy * fx; ++i)
            {
                weight_smem[vec + i] = p_wei[vec + i]
            }
        }
    }

    const int ih_base = (ho_tile_idx * TILE_OH) * sy - py;
    const int iw_base = (wo_tile_idx * TILE_OW) * sx - px;

    for(int vec = threadIdx.x * ELEMENTS_PER_THREAD; vec < H_in_tile * W_in_tile;
        vec += (blockDim.x * ELEMENTS_PER_THREAD))
    {
        // 解析坐标
        int tile_ih = vec / W_in_tile;
        int tile_iw = vec % W_in_tile;

        // 全局坐标(vec)
        // int ih_idx = ih_base + tile_ih;
        // int iw_idx = iw_base + tile_iw;

        floatxN in_val;
        for(int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            int cur_vec = vec + i;
            if(cur_vec < H_in_tile * W_in_tile)
            {
                int cur_tile_h = cur_vec / W_in_tile;
                int cur_tile_w = cur_vec % W_in_tile;
                int cur_ih     = ih_base + cur_tile_h;
                int cur_iw     = iw_base + cur_tile_w;

                if(cur_ih >= 0 && cur_ih < hi && cur_iw >= 0 && cur_iw < wi)
                {
                    int in_idx = batch_idx * hi * wi + cur_ih * wi + cur_iw;
                    in_val[i]  = p_in[in_idx];
                }
                else
                {
                    in_val[i] = 0.0f;
                }
            }
        }

        if(vec + ELEMENTS_PER_THREAD <= H_in_tile * W_in_tile)
        {
            *reinterpret_cast<floatxN*>(in_tile_smem + vec) = in_val;
        }
        else
        {
            for(int i = 0; i < ELEMENTS_PER_THREAD && (vec + i) < H_in_tile * W_in_tile; i++)
            {
                in_tile_smem[vec + i] = in_val[i];
            }
        }
    }

    __syncthreads();

    int tile_output_size = TILE_OH * TILE_OW;

    int tile_output_size    = TILE_OH * TILE_OW;
    int thread_output_count = (tile_output_size + blockDim.x * ELEMENTS_PER_THREAD - 1) /
                              (blockDim.x * ELEMENTS_PER_THREAD);

    /// 每个线程计算多个输出点
    for(int thread_vec = 0; thread_vec < thread_output_count; thread_vec++)
    {
        int start_vec =
            threadIdx.x * ELEMENTS_PER_THREAD + thread_vec * blockDim.x * ELEMENTS_PER_THREAD;

        floatxN output_val;

        for(int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            int linear_idx = start_vec + i;
            if(linear_idx < tile_output_size)
            {
                int oh_local = linear_idx / TILE_OW;
                int ow_local = linear_idx % TILE_OW;

                // 计算全局输出坐标
                int oh = ho_tile_idx * TILE_OH + oh_local;
                int ow = wo_tile_idx * TILE_OW + ow_local;

                if(oh < ho && ow < wo)
                {
                    float sum = 0.0f;

                    // 卷积计算
                    for(int fy_idx = 0; fy_idx < fy; fy_idx++)
                    {
                        for(int fx_idx = 0; fx_idx < fx; fx_idx++)
                        {
                            int ih = oh * sy - py + fy_idx * dy;
                            int iw = ow * sx - px + fx_idx * dx;

                            // 计算在输入tile中的位置
                            int tile_h = ih - ih_base;
                            int tile_w = iw - iw_base;

                            if(tile_h >= 0 && tile_h < H_in_tile && tile_w >= 0 &&
                               tile_w < W_in_tile)
                            {
                                float input_val  = in_tile_smem[tile_h * W_in_tile + tile_w];
                                float weight_val = weight_smem[fy_idx * fx + fx_idx];
                                sum += input_val * weight_val;
                            }
                        }
                    }

                    output_val[i] = sum;
                }
                else
                {
                    output_val[i] = 0.0f;
                }
            }
            else
            {
                output_val[i] = 0.0f;
            }
        }

        /// 写回全局内存
        for(int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            int linear_idx = start_vec + i;
            if(linear_idx < tile_output_size)
            {
                int oh_local = linear_idx / TILE_OW;
                int ow_local = linear_idx % TILE_OW;

                int oh = ho_tile_idx * TILE_OH + oh_local;
                int ow = wo_tile_idx * TILE_OW + ow_local;

                if(oh < ho && ow < wo)
                {
                    int out_idx    = batch_idx * ho * wo + oh * wo + ow;
                    p_out[out_idx] = output_val[i];
                }
            }
        }
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
                                int input_idx  = in_idx * c * hi * wi + ic * hi * wi + ih * wi + iw;
                                int weight_idx = ic * fy * fx + fh * fx + fw;
                                sum += p_in[input_idx] * p_wei[weight_idx];
                            }
                        }
                    }

                    int output_idx    = in_idx * c * ho * wo + ic * ho * wo + oh * wo + ow;
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
    int device_id = 0;
    hipSetDevice(device_id);

    int N = 1, C = 301500, H = 1, W = 53;
    int R = 1, S = 53;
    int pad_h = 0, pad_w = 0;
    int stride_h = 1, stride_w = 1;
    int dilate_h = 1, dilate_w = 1;
    int K = C;

    N = 1500, C = 1, H = 91, W = 91;
    R = 39, S = 39;
    pad_h = 0, pad_w = 0;
    stride_h = 1, stride_w = 1;
    dilate_h = 1, dilate_w = 1;
    K = C;

    // N = 1, C = 12, H = 1, W = 4;
    // R = 1, S = 4;
    // pad_h = 0, pad_w = 0;
    // stride_h = 1, stride_w = 1;
    // dilate_h = 1, dilate_w = 1;
    // K = C;

    int outH = (H + 2 * pad_h - dilate_h * (R - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilate_w * (S - 1) - 1) / stride_w + 1;

    // --------------------------------------------
    /// 入参限制条件
    /// C == K == group == 1
    // --------------------------------------------

    if(C != K || C != 1)
    {
        std::cout << "入参错误!!!检查入参限制条件,该kernel特定优化" << std::endl;
        return 1;
    }

    std::cout << "output size: " << outH << ", " << outW << std::endl;
    std::cout << "\n";

    size_t input_size  = N * C * H * W;
    size_t weight_size = K * R * S;
    size_t output_size = N * K * outH * outW;

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

    const int H_in_tile = tile_in_extent(TILE_OH, stride_h, dilate_h, R);
    const int W_in_tile = tile_in_extent(TILE_OW, stride_w, dilate_w, S);

    // 计算动态LDS大小
    int in_sublock_size  = H_in_tile * W_in_tile;
    int weight_size      = R * S;
    int dynamic_lds_size = (in_sublock_size + weight_size) * sizeof(float);

    int split_h = (outH + TILE_OH - 1) / TILE_OH;
    int split_w = (outW + TILE_OW - 1) / TILE_OW;

    unsigned int numgroupZ = ((N + BATCHS_SINGLE_BLOCK - 1) / (BATCHS_SINGLE_BLOCK));

    dim3 grid_dim(split_w, split_h, numgroupZ);
    dim3 block_dim(threadInBlock, 1, 1);

    std::cout << "blockDim(" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")"
              << "\n"
              << "gridDim(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")"
              << std::endl;

    // -----------------
    /// CPU验证
    // -----------------
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
    param.split_h   = split_h;
    param.split_w   = split_w;

    hipLaunchKernelGGL(depthwise_fwd, grid_dim, block_dim, dynamic_lds_size, 0, param);

    std::vector<float> outputGPUHostTensor(output_size);
    hipMemcpy(
        outputGPUHostTensor.data(), d_output, output_size * sizeof(float), hipMemcpyDeviceToHost);

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

#ifdef VERIFY
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

    printf("kernel time(ms): %7.5f\n", total_time / 100);
#endif

    hipFree(d_input);
    hipFree(d_weight);
    hipFree(d_output);

    return 0;
}
