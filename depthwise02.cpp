#include <hip/hip_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#define VERIFY 0
typedef _Float16 DATA_T;
#define ELEMENTS_PER_THREAD 8
#define OUTPUTS_PER_THREAD 4

typedef long bufferResourceDesc __attribute__((ext_vector_type(2)));

using StoreT  = DATA_T __attribute__((ext_vector_type(OUTPUTS_PER_THREAD)));
using AccessT = DATA_T __attribute__((ext_vector_type(ELEMENTS_PER_THREAD)));
using DotTx10 = _Float16 __attribute__((ext_vector_type(10)));
using FP16x2  = _Float16 __attribute__((ext_vector_type(2)));
using FP16x8  = _Float16 __attribute__((ext_vector_type(8)));
using uintx4  = unsigned int __attribute__((ext_vector_type(4)));

#define BATCHS_SINGLE_BLOCK 1
#define GROUP_SINGLE_BLOCK 16
#define TILE_OH 10
#define TILE_OW 10
#define THREADSBLOCK 256

struct DepthWiseParam
{
    void* __restrict__ inPtr;
    void* __restrict__ weiPtr;
    void* __restrict__ outPtr;
    int batch;
    int ih;
    int iw;
    int fy;
    int fx;
    int sy;
    int sx;
    int py;
    int px;
    int dy;
    int dx;
    int group;
    int oh;
    int ow;
    int split_h;
    int split_w;
    int tile_ih;
    int tile_iw;
};

template <typename T>
__device__ __forceinline__ bool in_bounds(T val, T low, T high)
{
    return val >= low && val < high;
}

typedef union Half2Float
{
    DotTx10 fp16x10 = {0.0f};
    struct FP16x2_DATA
    {
        FP16x2 x00;
        FP16x2 x01;
        FP16x2 x02;
        FP16x2 x03;
        FP16x2 x04;
    } FP16x2_DATA;
} Half2Float;

union val_t
{
    _Float16 h[8];
    uintx4 i;
};

__host__ __device__ __forceinline__ int tile_in_extent(int tile_o, int stride, int dilate, int RorS)
{
    // 覆盖 oh in [0..tile_o-1] 与 r in [0..R-1]:
    // ih spans: oh*stride + r*dilate  => size = (tile_o-1)*stride + (R-1)*dilate + 1
    return (tile_o - 1) * stride + (RorS - 1) * dilate + 1;
}

__device__ __forceinline__ void bufferLoadDwordx4(uintx4& data, int& index, bufferResourceDesc desc)
{
    asm volatile("buffer_load_dwordx4 %0, %1, %2, 0, idxen \n\t"
                 "s_waitcnt vmcnt(0) \n\t"
                 : "=v"(data), "+v"(index), "+s"(desc));
}

__global__ void fwd_depthwise(DepthWiseParam param)
{
    DATA_T* p_in  = (DATA_T*)param.inPtr;
    DATA_T* p_wei = (DATA_T*)param.weiPtr;
    DATA_T* p_out = (DATA_T*)param.outPtr;

    bufferResourceDesc desc_input;
    desc_input.x = (long)p_in;
    desc_input.x = (desc_input.x | (((long)(0x2 << 16)) << 32));
    desc_input.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));
    bufferResourceDesc desc_weight;
    desc_weight.x = (long)p_wei;
    desc_weight.x = (desc_weight.x | (((long)(0x2 << 16)) << 32));
    desc_weight.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));

    int batch   = param.batch;
    int ih      = param.ih;
    int iw      = param.iw;
    int fy      = param.fy;
    int fx      = param.fx;
    int sy      = param.sy;
    int sx      = param.sx;
    int py      = param.py;
    int px      = param.px;
    int dy      = param.dy;
    int dx      = param.dx;
    int group   = param.group;
    int oh      = param.oh;
    int ow      = param.ow;
    int split_h = param.split_h;
    int split_w = param.split_w;
    // int tile_ih = param.tile_ih;
    // int tile_iw = param.tile_iw;
    // ===========================
    /// 针对3x3; group % 8 == 0
    // ===========================

    int ohw_tile_idx = blockIdx.y;

    int wo_tile_idx = ohw_tile_idx % split_w;
    int ho_tile_idx = ohw_tile_idx / split_w;

    int group_base = blockIdx.x * GROUP_SINGLE_BLOCK;
    int batch_id   = blockIdx.z;

    int oh_start         = ho_tile_idx * TILE_OH;
    int ow_start         = wo_tile_idx * TILE_OW;
    int group_start      = group_base;
    int tile_oh          = min(TILE_OH, oh - oh_start);
    int tile_ow          = min(TILE_OW, ow - ow_start);
    int group_block_deal = min(GROUP_SINGLE_BLOCK, group - group_start);

    int tile_ih = tile_in_extent(tile_oh, sy, dy, fy);
    int tile_iw = tile_in_extent(tile_ow, sx, dx, fx);

    extern __shared__ DATA_T smem[];
    DATA_T* weight_smem = smem;
    DATA_T* in_smem     = weight_smem + GROUP_SINGLE_BLOCK * fy * fx;

    int in_block_deal     = group_block_deal * tile_ih * tile_iw;
    int weight_block_deal = group_block_deal * fx * fy;

    int ih_base = oh_start * sy - py;
    int iw_base = ow_start * sx - px;

    uintx4 value = {0};
    for(int vec = threadIdx.x * ELEMENTS_PER_THREAD; vec < in_block_deal;
        vec += (blockDim.x * ELEMENTS_PER_THREAD))
    {
        int ih_in_tile = ((vec / group_block_deal) / tile_iw) % tile_ih;
        int iw_in_tile = (vec / group_block_deal) % tile_iw;
        int kg_in_tile = vec % group_block_deal;

        int ih_idx = ih_in_tile + ih_base;    // 不会越界 边界判断重新审视
        int iw_idx = iw_in_tile + iw_base;    // 不会越界 边界判断重新审视
        int kg_idx = kg_in_tile + group_base; // 可能越界，向量化的导致的

        bool flag =
            in_bounds(ih_idx, 0, ih) && in_bounds(iw_idx, 0, iw) && in_bounds(kg_idx, 0, group);

        int idx = flag
                      ? (batch_id * ih * iw * group + ih_idx * iw * group + iw_idx * group + kg_idx)
                      : -1;

        // printf("tid=%d, flag=%d ih_idx=%d idx=%d\n", threadIdx.x, flag, ih_idx, idx);

        bufferLoadDwordx4(value, idx, desc_input);
        // val_t i2h;
        // i2h.i = value;

        // printf("tid=%d, %f, %f, %f, %f, %f, %f, %f, %f\n",
        //        threadIdx.x,
        //        (float)i2h.h[0],
        //        (float)i2h.h[1],
        //        (float)i2h.h[2],
        //        (float)i2h.h[3],
        //        (float)i2h.h[4],
        //        (float)i2h.h[5],
        //        (float)i2h.h[6],
        //        (float)i2h.h[7]);

        int smem_store_idx =
            ih_in_tile * tile_iw * group_block_deal + iw_in_tile * group_block_deal + kg_in_tile;
        *((uintx4*)(&in_smem[smem_store_idx])) = value;
    }

    for(int tid = threadIdx.x; tid < weight_block_deal; tid += blockDim.x)
    {
        int kfy          = (tid / fx) % fy;
        int kfx          = tid % fx;
        int kg           = tid / (fx * fy);
        weight_smem[tid] = p_wei[(group_base + kg) * fy * fx + kfy * fx + kfx];
    }

    __syncthreads();

    int total_outputs_in_tile = tile_oh * tile_ow * group_block_deal;
    for(int out_idx_in_tile = threadIdx.x; out_idx_in_tile < total_outputs_in_tile;
        out_idx_in_tile += blockDim.x)
    {
        int oh_in_tile = (out_idx_in_tile / group_block_deal / tile_ow) % tile_oh;
        int ow_in_tile = (out_idx_in_tile / group_block_deal) % tile_ow;
        int kg_in_tile = out_idx_in_tile % group_block_deal;

        int oh_global = ho_tile_idx * TILE_OH + oh_in_tile;
        int ow_global = wo_tile_idx * TILE_OW + ow_in_tile;
        int kg_global = kg_in_tile + group_base;

        if(oh_global >= oh || ow_global >= ow || kg_global >= group)
            continue;

        float acc[GROUP_SINGLE_BLOCK] = {0.0f};
        Half2Float in_map             = {0.0f};
        Half2Float weight             = {0.0f};

        int ih_in_tile0 = oh_in_tile * sy + 0 * dy;
        int ih_in_tile1 = oh_in_tile * sy + 1 * dy;
        int ih_in_tile2 = oh_in_tile * sy + 2 * dy;

        int iw_in_tile0 = ow_in_tile * sx + 0 * dx;
        int iw_in_tile1 = ow_in_tile * sx + 1 * dx;
        int iw_in_tile2 = ow_in_tile * sx + 2 * dx;

        in_map.fp16x10[0] = in_smem[ih_in_tile0 * tile_iw * group_block_deal +
                                    iw_in_tile0 * group_block_deal + kg_in_tile];
        in_map.fp16x10[1] = in_smem[ih_in_tile0 * tile_iw * group_block_deal +
                                    iw_in_tile1 * group_block_deal + kg_in_tile];
        in_map.fp16x10[2] = in_smem[ih_in_tile0 * tile_iw * group_block_deal +
                                    iw_in_tile2 * group_block_deal + kg_in_tile];
        in_map.fp16x10[3] = in_smem[ih_in_tile1 * tile_iw * group_block_deal +
                                    iw_in_tile0 * group_block_deal + kg_in_tile];
        in_map.fp16x10[4] = in_smem[ih_in_tile1 * tile_iw * group_block_deal +
                                    iw_in_tile1 * group_block_deal + kg_in_tile];
        in_map.fp16x10[5] = in_smem[ih_in_tile1 * tile_iw * group_block_deal +
                                    iw_in_tile2 * group_block_deal + kg_in_tile];
        in_map.fp16x10[6] = in_smem[ih_in_tile2 * tile_iw * group_block_deal +
                                    iw_in_tile0 * group_block_deal + kg_in_tile];
        in_map.fp16x10[7] = in_smem[ih_in_tile2 * tile_iw * group_block_deal +
                                    iw_in_tile1 * group_block_deal + kg_in_tile];
        in_map.fp16x10[8] = in_smem[ih_in_tile2 * tile_iw * group_block_deal +
                                    iw_in_tile2 * group_block_deal + kg_in_tile];

        weight.fp16x10[0] = weight_smem[kg_in_tile * fy * fx + 0];
        weight.fp16x10[1] = weight_smem[kg_in_tile * fy * fx + 1];
        weight.fp16x10[2] = weight_smem[kg_in_tile * fy * fx + 2];
        weight.fp16x10[3] = weight_smem[kg_in_tile * fy * fx + 3];
        weight.fp16x10[4] = weight_smem[kg_in_tile * fy * fx + 4];
        weight.fp16x10[5] = weight_smem[kg_in_tile * fy * fx + 5];
        weight.fp16x10[6] = weight_smem[kg_in_tile * fy * fx + 6];
        weight.fp16x10[7] = weight_smem[kg_in_tile * fy * fx + 7];
        weight.fp16x10[8] = weight_smem[kg_in_tile * fy * fx + 8];

        acc[kg_in_tile] = __builtin_amdgcn_fdot2(
            in_map.FP16x2_DATA.x00, weight.FP16x2_DATA.x00, acc[kg_in_tile], false);
        acc[kg_in_tile] = __builtin_amdgcn_fdot2(
            in_map.FP16x2_DATA.x01, weight.FP16x2_DATA.x01, acc[kg_in_tile], false);
        acc[kg_in_tile] = __builtin_amdgcn_fdot2(
            in_map.FP16x2_DATA.x02, weight.FP16x2_DATA.x02, acc[kg_in_tile], false);
        acc[kg_in_tile] = __builtin_amdgcn_fdot2(
            in_map.FP16x2_DATA.x03, weight.FP16x2_DATA.x03, acc[kg_in_tile], false);
        // acc = __builtin_amdgcn_fdot2(
        //     in_map.FP16x2_DATA.x04, weight.FP16x2_DATA.x04, acc, false);
        acc[kg_in_tile] += in_map.fp16x10[8] * weight.fp16x10[8];

        int out_offset =
            batch_id * oh * ow * group + oh_global * ow * group + ow_global * group + kg_global;
        p_out[out_offset] = static_cast<DATA_T>(acc[kg_in_tile]);
    }
}

// ------------------------------
/// Host-side
// ------------------------------
void fwd_depthwise_nhwc_cpu(DATA_T* input,
                            DATA_T* weight,
                            DATA_T* output,
                            int batch,
                            int ih,
                            int iw,
                            int fh,
                            int fw,
                            int sh,
                            int sw,
                            int ph,
                            int pw,
                            int dh,
                            int dw,
                            int group,
                            int oh,
                            int ow)
{
    for(int kb = 0; kb < batch; ++kb)
    {

        for(int koh = 0; koh < oh; ++koh)
        {
            for(int kow = 0; kow < ow; ++kow)
            {
                for(int kg = 0; kg < group; ++kg)
                {
                    float sum = 0.0f;
                    for(int kfh = 0; kfh < fh; ++kfh)
                    {
                        for(int kfw = 0; kfw < fw; ++kfw)
                        {
                            int ih_mapcoord = koh * sh - ph + kfh * dh;
                            int iw_mapcoord = kow * sw - pw + kfw * dw;

                            if(ih_mapcoord >= 0 && ih_mapcoord < ih && iw_mapcoord >= 0 &&
                               iw_mapcoord < iw)
                            {
                                size_t input_idx =
                                    ((kb * ih + ih_mapcoord) * iw + iw_mapcoord) * group + kg;
                                size_t weight_idx = (kg * fh + kfh) * fw + kfw;

                                sum += static_cast<float>(input[input_idx] * weight[weight_idx]);
                            }
                        }
                    }
                    size_t output_idx  = ((kb * oh + koh) * ow + kow) * group + kg;
                    output[output_idx] = static_cast<DATA_T>(sum);
                }
            }
        }
    }
}

#define COLOR_RED "\033[1;31m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_BLUE "\033[1;34m"
#define COLOR_RESET "\033[0m"

bool compare_arrary(const DATA_T* cpu_output,
                    const DATA_T* gpu_output,
                    size_t size,
                    float tolerance = 1e-2f)
{
    for(size_t i = 0; i < size; ++i)
    {
        float a = static_cast<float>(cpu_output[i]);
        float b = static_cast<float>(gpu_output[i]);
        if(std::abs(a - b) > tolerance)
        {
            std::cout << COLOR_RED << "\nMismatch at idx " << i << ": " << a << " vs " << b
                      << " (diff: " << std::abs(a - b) << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T>
void print_matrix(T* data, int len, int start, int stop)
{
    for(int i = start; i < std::min(len, stop); ++i)
    {
        printf("%6.5f ", static_cast<float>(data[i]));
        if(i % 32 == 31)
        {
            printf("\n");
        }
    }
    return;
}

float rand_float(float min = -1.0f, float max = 1.0f)
{
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (max - min));
}

int main(int argc, char** argv)
{
    int batch = 250; // atoi(argv[1]);
    int ih    = 80; // atoi(argv[2]);
    int iw    = 80; // atoi(argv[3]);
    int fy    = 3; // atoi(argv[4]);
    int fx    = 3; // atoi(argv[5]);
    int sy    = 1; // atoi(argv[6]);
    int sx    = 1; // atoi(argv[7]);
    int py    = 1; // atoi(argv[8]);
    int px    = 1; // atoi(argv[9]);
    int dy    = 1; // atoi(argv[10]);
    int dx    = 1; // atoi(argv[11]);
    int group = 256; // atoi(argv[12]);
    int c;
    int k = c = group;

    int oh = (ih + 2 * py - dy * (fy - 1) - 1) / sy + 1;
    int ow = (iw + 2 * px - dx * (fx - 1) - 1) / sx + 1;

    std::cout << "ouput spatial_dim: " << oh << ", " << ow << std::endl;

    int input_size  = batch * group * ih * iw;
    int weight_size = group * fy * fx;
    int output_size = batch * group * oh * ow;

    std::vector<DATA_T> h_input(input_size);
    std::vector<DATA_T> h_weight(weight_size);
    std::vector<DATA_T> h_output(output_size, 0.0f);

    for(size_t i = 0; i < h_input.size(); ++i)
    {
        h_input[i] = static_cast<DATA_T>(rand_float());
    }

    for(size_t i = 0; i < h_weight.size(); ++i)
    {
        h_weight[i] = static_cast<DATA_T>(rand_float());
    }

    DATA_T *d_input, *d_weight, *d_output;
    hipMalloc(&d_input, sizeof(DATA_T) * input_size);
    hipMalloc(&d_weight, sizeof(DATA_T) * weight_size);
    hipMalloc(&d_output, sizeof(DATA_T) * output_size);

    hipMemcpy(d_input, h_input.data(), sizeof(DATA_T) * input_size, hipMemcpyHostToDevice);
    hipMemcpy(d_weight, h_weight.data(), sizeof(DATA_T) * weight_size, hipMemcpyHostToDevice);
    hipMemset(d_output, 0, sizeof(DATA_T) * output_size);

    const int tile_ih = tile_in_extent(TILE_OH, sy, dy, fy);
    const int tile_iw = tile_in_extent(TILE_OW, sx, dx, fx);

    int in_sublock_size     = GROUP_SINGLE_BLOCK * tile_ih * tile_iw;
    int weight_single_block = GROUP_SINGLE_BLOCK * fy * fx;
    int dynamic_lds_bytes   = sizeof(DATA_T) * (in_sublock_size + weight_single_block);

    std::cout << "dynamic_lds_bytes: " << dynamic_lds_bytes << std::endl;

    int split_h = (oh + TILE_OH - 1) / TILE_OH;
    int split_w = (ow + TILE_OW - 1) / TILE_OW;

    int grid_x = (group + GROUP_SINGLE_BLOCK - 1) / GROUP_SINGLE_BLOCK;
    int grid_y = split_h * split_w;
    int grid_z = batch;

    dim3 grid_dim(grid_x, grid_y, grid_z);
    dim3 block_dim(THREADSBLOCK, 1, 1);

    std::cout << "grid(" << grid_x << ", " << grid_y << ", " << grid_z << ")" << "\n"
              << "block(" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")"
              << std::endl;

    DepthWiseParam param;
    param.inPtr   = d_input;
    param.weiPtr  = d_weight;
    param.outPtr  = d_output;
    param.batch   = batch;
    param.ih      = ih;
    param.iw      = iw;
    param.fy      = fy;
    param.fx      = fx;
    param.sy      = sy;
    param.sx      = sx;
    param.py      = py;
    param.px      = px;
    param.dy      = dy;
    param.dx      = dx;
    param.group   = group;
    param.oh      = oh;
    param.ow      = ow;
    param.split_h = split_h;
    param.split_w = split_w;
    param.tile_ih = tile_ih;
    param.tile_iw = tile_iw;

    hipLaunchKernelGGL(fwd_depthwise, grid_dim, block_dim, dynamic_lds_bytes, 0, param);

    if(VERIFY)
    {
        std::vector<DATA_T> outputGpuHostTensor(output_size);
        hipMemcpy(outputGpuHostTensor.data(),
                  d_output,
                  sizeof(DATA_T) * output_size,
                  hipMemcpyDeviceToHost);

        fwd_depthwise_nhwc_cpu(h_input.data(),
                               h_weight.data(),
                               h_output.data(),
                               batch,
                               ih,
                               iw,
                               fy,
                               fx,
                               sy,
                               sx,
                               py,
                               px,
                               dy,
                               dx,
                               group,
                               oh,
                               ow);

        std::cout << "\n================== HOST   result ==================" << std::endl;
        print_matrix(h_output.data(), h_output.size(), 0, 128);
        std::cout << "\n================== DEVICE result ==================" << std::endl;
        print_matrix(outputGpuHostTensor.data(), outputGpuHostTensor.size(), 0, 128);

        bool match =
            compare_arrary(h_output.data(), outputGpuHostTensor.data(), outputGpuHostTensor.size());
        if(match)
        {
            std::cout << COLOR_GREEN << "\nSUCCESS" << COLOR_RESET << std::endl;
        }
        else
        {
            std::cout << COLOR_RED << "\nFAILURE" << COLOR_RESET << std::endl;
        }
    }
    else
    {
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        int iteration  = 100;
        float cur_time = 0.0f, total_time = 0.0f;
        for(int iter = 0; iter < iteration; ++iter)
        {
            hipEventRecord(start, 0);
            hipLaunchKernelGGL(fwd_depthwise, grid_dim, block_dim, dynamic_lds_bytes, 0, param);
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&cur_time, start, stop);
            total_time += cur_time;
        }

        float avg_time_ms = total_time / iteration;
        std::cout << "kernel time(ms): " << COLOR_BLUE << avg_time_ms << COLOR_RESET << std::endl;

        /// KERNEL分析
        float kernel_time_s    = avg_time_ms * 1e-3;
        size_t actual_io_bytes = sizeof(DATA_T) * (input_size + weight_size + output_size);
        float actual_io_gb     = actual_io_bytes * 1e-9;
        float bandwidth_gbs    = actual_io_gb / kernel_time_s;

        std::cout << "\nkernel_time(s): " << kernel_time_s << " s"
                  << "\nactual_io_gb: " << actual_io_gb << " GB"
                  << "\nbandwidth_gbs: " << bandwidth_gbs << " GB/s\n";

        float theoretical_bandwidth_gbs = 1.5 * 1e3; // 1.5TB/s
        float bandwidth_utilization     = bandwidth_gbs / theoretical_bandwidth_gbs;

        std::cout << "\nbandwidth_utilization: " << COLOR_BLUE << bandwidth_utilization * 100
                  << " %" << COLOR_RESET << std::endl;

        /// 性能瓶颈分析
        size_t total_flops         = output_size * fy * fx * 2;
        float arithmetic_intensity = (float)(total_flops) / (bandwidth_gbs * 1e9);

        ///
        float peak_flops_TFLOPS = 7.3f;
        float peak_flops_FLOPS  = 7.3 * 1e3;
        float op_bytes_ratio    = peak_flops_FLOPS / (theoretical_bandwidth_gbs * 1e9);

        std::cout << "\narithmetic_intensity = " << arithmetic_intensity
                  << " flops/Bytes(ops/bytes)"
                  << "\nop_bytes_ratio = " << op_bytes_ratio << " flops/Bytes(BW_math/BW_mem)"
                  << std::endl;

        if(arithmetic_intensity > op_bytes_ratio)
        {
            std::cout << "math limit!" << std::endl;
        }
        else
        {
            std::cout << "memory limit!" << std::endl;
        }
    }

    hipFree(d_input);
    hipFree(d_weight);
    hipFree(d_output);
}
