#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include <fstream>
#include <hip/hip_fp16.h>

typedef _Float16 DATA_TYPE;

#define C_PER_BLK (64)
#define NPQ_PER_BLK (64)
#define ITER_DEPTH (32)

#define VERIFY (1)
#define PERFORM_MESURE_TIMES (100)

#define DIV_CEIL(M, BM) (((M)-1) / (BM) + 1)
#define LESS_U32(x, high) ((unsigned int)x < (unsigned int)high)
#define MIN(x, y) ((x) > (y) ? (y) : (x))

#define SHARED_MEM __attribute__((address_space(3)))

typedef float __attribute__((ext_vector_type(16))) fp32x16;
typedef float __attribute__((ext_vector_type(8))) fp32x8;
// typedef _Float16 __attribute__((ext_vector_type(8))) fp16x8;
typedef float __attribute__((ext_vector_type(2))) fp32x2;

typedef _Float16 __attribute__((ext_vector_type(8))) fp16x8;
typedef _Float16 __attribute__((ext_vector_type(4))) fp16x4;

typedef int32_t __attribute__((ext_vector_type(16))) i32x16;
typedef int32_t __attribute__((ext_vector_type(8))) i32x8;
typedef int32_t __attribute__((ext_vector_type(4))) i32x4;
typedef int32_t __attribute__((ext_vector_type(2))) i32x2;

typedef union
{
    fp16x8 fp16x8_data;
    i32x4 i32x4_data;
} GloadUnion;

typedef union
{
    fp16x2 fp32x2_data;
    unsigned int i32x2_data;
} GloadDWordx2Union;

typedef union
{
    struct
    {
        unsigned long addr_base;
        int record_nums;
        int config;
    } desc_data;
    i32x4 i32x4_data;
} BufDescUnion;

typedef union
{
    fp16x8 vec4_data;
    struct
    {
        fp16x4 front;
        fp16x4 rear;
    } vec2_data;
} CalUnion;

typedef struct OffsetIgemmParam_
{
    void* __restrict__ grad_output;
    void* __restrict__ weight;
    void* __restrict__ data_im;
    void* __restrict__ data_offset;
    void* __restrict__ data_mask;
    void* __restrict__ grad_offset;
    void* __restrict__ grad_mask;

    int batch_size;
    int channels;
    int height;
    int width;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int group;
    int deformable_group;

    int out_h;
    int out_w;
} OffsetIgemmParam;

template <typename T>
void print_matrix(T* data, size_t len, size_t start, size_t end)
{
    for(size_t i = start; i < std::min(len, end); i++)
    {
        printf("%6.5f, ", static_cast<float>(data[i]));
        if(i % 32 == 31)
        {
            printf("\n");
        }
    }

    return;
}

#define HIP_CHECK(error)                                                                  \
    if(error != hipSuccess)                                                               \
    {                                                                                     \
        std::cout << "HIP error: " << hipGetErrorString(error) << " at line " << __LINE__ \
                  << std::endl;                                                           \
        exit(1);                                                                          \
    }

#define HIP_1D_KERNEL_LOOP(i, n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ void igemm_dmcn_get_coordinate_weight_nhwc(fp32x16& weight,
                                                      float argmax_h,
                                                      float argmax_w,
                                                      const int batch_size,
                                                      const int height,
                                                      const int width,
                                                      const int channels,
                                                      const i32x16 c_index,
                                                      const int n_index,
                                                      const float* im_data,
                                                      const int bp_dir)
{
#pragma unroll
    for(int i = 0; i < 16; i++)
    {
        weight[i] = 0.0f;
    }

    if(n_index >= batch_size || argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
       argmax_w >= width)
    {
        return;
    }

    int argmax_h_low  = floorf(argmax_h);
    int argmax_w_low  = floorf(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    int n_stride = height * width * channels;

    // NHWC索引计算: (h * width + w) * channels + c
    if(bp_dir == 0) // 计算x方向的梯度（即对Δx的梯度）
    {
        if(argmax_h_low >= 0 && argmax_w_low >= 0)
        {
            int base_pose = n_index * n_stride + (argmax_h_low * width + argmax_w_low) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += -1 * (argmax_w_low + 1 - argmax_w) * im_data[base_pose + c_index[i]];
            }
        }

        if(argmax_h_low >= 0 && argmax_w_high <= width - 1)
        {
            int base_pose = n_index * n_stride + (argmax_h_low * width + argmax_w_high) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += -1 * (argmax_w - argmax_w_low) * im_data[base_pose + c_index[i]];
            }
        }

        if(argmax_h_high <= height - 1 && argmax_w_low >= 0)
        {
            int base_pose = n_index * n_stride + (argmax_h_high * width + argmax_w_low) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += (argmax_w_low + 1 - argmax_w) * im_data[base_pose + c_index[i]];
            }
        }

        if(argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        {
            int base_pose = n_index * n_stride + (argmax_h_high * width + argmax_w_high) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += (argmax_w - argmax_w_low) * im_data[base_pose + c_index[i]];
            }
        }
    }
    else if(bp_dir == 1) // 计算y方向的梯度（即对Δy的梯度）
    {
        if(argmax_h_low >= 0 && argmax_w_low >= 0)
        {
            int base_pose = n_index * n_stride + (argmax_h_low * width + argmax_w_low) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += -1 * (argmax_h_low + 1 - argmax_h) * im_data[base_pose + c_index[i]];
            }
        }

        if(argmax_h_low >= 0 && argmax_w_high <= width - 1)
        {
            int base_pos = n_index * n_stride + (argmax_h_low * width + argmax_w_high) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += (argmax_h_low + 1 - argmax_h) * im_data[base_pos + c_index[i]];
            }
        }

        if(argmax_h_high <= height - 1 && argmax_w_low >= 0)
        {
            int base_pos = n_index * n_stride + (argmax_h_high * width + argmax_w_low) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += -1 * (argmax_h - argmax_h_low) * im_data[base_pos + c_index[i]];
            }
        }

        if(argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        {
            int base_pos = n_index * n_stride + (argmax_h_high * width + argmax_w_high) * channels;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                weight[i] += (argmax_h - argmax_h_low) * im_data[base_pos + c_index[i]];
            }
        }
    }

    return;
}

__device__ float sum_fp32_cross_lane_stride16(const float val, const int bpermute_addr)
{
    float sum = val;
    sum += __hip_ds_swizzlef_N<0x401F>(sum);       // tid0:0+16 tid32: 32+48
    sum += __hip_ds_bpermutef(bpermute_addr, sum); // tid0:0+16+32+48
    return sum;
}

__global__ void igemm_offset_fp32_nhwc(OffsetIgemmParam param)
{
    DATA_TYPE* grad_output = (DATA_TYPE*)param.grad_output;
    DATA_TYPE* weight      = (DATA_TYPE*)param.weight;
    DATA_TYPE* data_im     = (DATA_TYPE*)param.data_im;
    DATA_TYPE* data_offset = (DATA_TYPE*)param.data_offset;
    DATA_TYPE* data_mask   = (DATA_TYPE*)param.data_mask;

    DATA_TYPE* grad_offset = (DATA_TYPE*)param.grad_offset;
    DATA_TYPE* grad_mask   = (DATA_TYPE*)param.grad_mask;

    int batch_size       = param.batch_size;
    int channels         = param.channels;
    int height           = param.height;
    int width            = param.width;
    int out_channels     = param.out_channels;
    int kernel_h         = param.kernel_h;
    int kernel_w         = param.kernel_w;
    int pad_h            = param.pad_h;
    int pad_w            = param.pad_w;
    int stride_h         = param.stride_h;
    int stride_w         = param.stride_w;
    int dilation_h       = param.dilation_h;
    int dilation_w       = param.dilation_w;
    int group            = param.group;
    int deformable_group = param.deformable_group;

    int out_h = param.out_h;
    int out_w = param.out_w;

    BufDescUnion out_desc;
    out_desc.desc_data.addr_base = (unsigned long)grad_output;
    out_desc.i32x4_data.y |= (2 << 16);
    out_desc.i32x4_data.z = 0xFFFFFFFE;
    out_desc.i32x4_data.w = 0x20000;

    BufDescUnion wei_desc;
    wei_desc.desc_data.addr_base = (unsigned long)weight;
    wei_desc.i32x4_data.y |= (2 << 16);
    wei_desc.i32x4_data.z = 0xFFFFFFFE;
    wei_desc.i32x4_data.w = 0x20000;

    BufDescUnion im_desc;
    im_desc.desc_data.addr_base = (unsigned long)data_im;
    im_desc.i32x4_data.y |= (2 << 16);
    im_desc.i32x4_data.z = 0xFFFFFFFE;
    im_desc.i32x4_data.w = 0x20000;

    BufDescUnion offset_desc;
    offset_desc.desc_data.addr_base = (unsigned long)data_offset;
    offset_desc.i32x4_data.y |= (2 << 16);
    offset_desc.i32x4_data.z = 0xFFFFFFFE;
    offset_desc.i32x4_data.w = 0x20000;

    BufDescUnion mask_desc;
    mask_desc.desc_data.addr_base = (unsigned long)data_mask;
    mask_desc.i32x4_data.y |= (2 << 16);
    mask_desc.i32x4_data.z = 0xFFFFFFFE;
    mask_desc.i32x4_data.w = 0x20000;

    BufDescUnion grad_offset_desc;
    grad_offset_desc.desc_data.addr_base = (unsigned long)grad_offset;
    grad_offset_desc.i32x4_data.y |= (2 << 16);
    grad_offset_desc.i32x4_data.z = 0xFFFFFFFE;
    grad_offset_desc.i32x4_data.w = 0x20000;

    __shared__ DATA_TYPE lds_out[64 * 32];
    __shared__ DATA_TYPE lds_wei[64 * 32];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bx  = blockIdx.x;
    int by  = blockIdx.y;
    int bz  = blockIdx.z;

    int lane_id = __lane_id();
    int wave_id = tid >> 6;
    wave_id     = __builtin_amdgcn_readfirstlane(wave_id);

    int bpermute_addr = (lane_id ^ 32) << 2;

    int blk_map_wei_c   = bx * C_PER_BLK;
    int blk_map_out_npq = by * NPQ_PER_BLK;

    int map_out_npq = blk_map_out_npq + wave_id * 16 + (lane_id >> 2); // 映射64个npq   4 * 16 = 64
    int map_out_k   = (lane_id & 3) * 8;                               // 4 * 8 = 32
    int valid_out_npq = LESS_U32(map_out_npq, batch_size * out_h * out_w);

    int map_wei_c   = blk_map_wei_c + (lane_id & 7) * 8; // 0-7 映射64个c
    int map_wei_k   = wave_id * 8 + (lane_id >> 3);
    int valid_wei_c = LESS_U32(map_wei_c, channels);

    // TODO: 现在是转置写入
    int lds_out_wr_pos = wave_id * 16 + (lane_id >> 3) + (lane_id & 3) * (8 * 64);
    int lds_out_rd_pos = (wave_id & 3) * 16 + (lane_id & 15) + (lane_id >> 4) * 2 * 64;

    int lds_wei_wr_pos = wave_id * (64 * 8) + (lane_id & 7) * 8 + (lane_id >> 3) * 64; // (8*8) * 8
    int lds_wei_rd_pos = (lane_id & 7) * 4 + (lane_id >> 3) * 64;

    GloadUnion g2s_out_0, g2s_out_1;
    GloadUnion g2s_wei_0, g2s_wei_1;

    CalUnion cal_out_0;
    CalUnion cal_wei_0;
    CalUnion cal_wei_1;

    int r_index = 0;
    int s_index = 0;
    for(int r_index = 0; r_index < kernel_h; r_index++)
    {
        for(int s_index = 0; s_index < kernel_w; s_index++)
        {
            fp32x4 res0 = {0.0f};
            fp32x4 res1 = {0.0f};
            fp32x4 res2 = {0.0f};
            fp32x4 res3 = {0.0f};

            // 每次加载32深度
            for(int k_start = 0; k_start < out_channels; k_start += ITER_DEPTH)
            {
                int load_out_k     = k_start + map_out_k;
                int load_out_pos   = map_out_npq * out_channels + load_out_k;
                bool valid_out_k   = LESS_U32(load_out_k, out_channels);
                int load_out_pos_0 = valid_out_npq && valid_out_k ? load_out_pos : -1;

                int load_wei_k   = k_start + map_wei_k;
                int load_wei_pos = load_wei_k * (kernel_h * kernel_w * channels) +
                                   (r_index * kernel_w + s_index) * channels + map_wei_c;
                bool valid_wei_k   = LESS_U32(load_wei_k, out_channels);
                int load_wei_pos_0 = valid_wei_c && valid_wei_k ? load_wei_pos : -1;

                g2s_out_0.i32x4_data = __builtin_amdgcn_buffer_load_dwordx4(
                    out_desc.i32x4_data, load_out_pos_0, 0, 0, 0);

                g2s_wei_0.i32x4_data = __builtin_amdgcn_buffer_load_dwordx4(
                    wei_desc.i32x4_data, load_wei_pos_0, 0, 0, 0);

                __syncthreads();

                lds_out[lds_out_wr_pos + 0 * 64] = g2s_out_0.fp16x8_data[0];
                lds_out[lds_out_wr_pos + 1 * 64] = g2s_out_0.fp16x8_data[1];
                lds_out[lds_out_wr_pos + 2 * 64] = g2s_out_0.fp16x8_data[2];
                lds_out[lds_out_wr_pos + 3 * 64] = g2s_out_0.fp16x8_data[3];

                lds_out[lds_out_wr_pos + 4 * 64] = g2s_out_0.fp16x8_data[4];
                lds_out[lds_out_wr_pos + 5 * 64] = g2s_out_0.fp16x8_data[5];
                lds_out[lds_out_wr_pos + 6 * 64] = g2s_out_0.fp16x8_data[6];
                lds_out[lds_out_wr_pos + 7 * 64] = g2s_out_0.fp16x8_data[7];

                *(fp16x8_data*)&lds_wei[lds_wei_wr_pos] = g2s_wei_0.fp16x8_data;

                __syncthreads();

                cal_out_0.vec2_data.front.x = lds_out[lds_out_rd_pos + 0 * 16 * 64 + 0 * 64];
                cal_out_0.vec2_data.front.y = lds_out[lds_out_rd_pos + 0 * 16 * 64 + 1 * 64];

                cal_wei_0.vec4_data = __builtin_amdgcn_ds_read_m32x16f16(
                    (SHARED_MEM DATA_TYPE*)&lds_wei[lds_wei_rd_pos + 0 * 8 * 64 + 0 * 32],
                    (short)0);
                cal_wei_1.vec4_data = __builtin_amdgcn_ds_read_m32x16f16(
                    (SHARED_MEM DATA_TYPE*)&lds_wei[lds_wei_rd_pos + 0 * 8 * 64 + 1 * 32],
                    (short)0);

                res0 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_0.vec2_data.front,
                                                             res0); // c: 0 4 8 12  npq: 0
                res1 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_0.vec2_data.rear,
                                                             res1); // c: 16 20 24 28  npq: 0
                res2 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_1.vec2_data.front,
                                                             res2); // c: 32 36 40 44  npq: 0
                res3 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_1.vec2_data.rear,
                                                             res3); // c: 48 52 56 60  npq: 0

                __syncthreads();

                cal_out_0.vec2_data.front.x = lds_out[lds_out_rd_pos + 1 * 16 * 64 + 0 * 64];
                cal_out_0.vec2_data.front.y = lds_out[lds_out_rd_pos + 1 * 16 * 64 + 1 * 64];

                cal_wei_0.vec4_data = __builtin_amdgcn_ds_read_m32x16f16(
                    (SHARED_MEM DATA_TYPE*)&lds_wei[lds_wei_rd_pos + 1 * 8 * 64 + 0 * 32],
                    (short)0);
                cal_wei_1.vec4_data = __builtin_amdgcn_ds_read_m32x16f16(
                    (SHARED_MEM DATA_TYPE*)&lds_wei[lds_wei_rd_pos + 1 * 8 * 64 + 1 * 32],
                    (short)0);

                res0 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_0.vec2_data.front,
                                                             res0); // c: 0 4 8 12  npq: 0
                res1 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_0.vec2_data.rear,
                                                             res1); // c: 16 20 24 28  npq: 0
                res2 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_1.vec2_data.front,
                                                             res2); // c: 32 36 40 44  npq: 0
                res3 = __builtin_amdgcn_mmac_f32_16x16x16f16(cal_out_0.vec2_data.front,
                                                             cal_wei_1.vec2_data.rear,
                                                             res3); // c: 48 52 56 60  npq: 0
            }

            __syncthreads();

            GloadDWordx2Union offset_val0;
            // 每个block计算出 64个 npq 因此需要加载64个npq对应的 im中的插值数据
            // 加载对应的mask的值 并进行双线性差值
            int mapped_c0_0 = blk_map_wei_c + (lane_id >> 4);
            int bp_dir      = 0;

            i32x16 mapped_c0;
#pragma unroll
            for(int i = 0; i < 16; i++)
            {
                mapped_c0[i] = mapped_c0_0 + i * 4;
            }

            int load_offset_npq_0 = blk_map_out_npq + (wave_id & 3) * 16 + (lane_id & 15);
            int load_offset_rs    = r_index * kernel_w + s_index;

            int mapped_n0      = load_offset_npq_0 / (out_h * out_w);
            int mapped_p0      = (load_offset_npq_0 / out_w) % out_h;
            int mapped_q0      = load_offset_npq_0 % out_w;
            int valid_offset_0 = LESS_U32(load_offset_npq_0, batch_size * out_h * out_w) &&
                                 LESS_U32(load_offset_rs, kernel_h * kernel_w);

            int load_offset_pos_0 =
                load_offset_npq_0 * (kernel_h * kernel_w * 2) + load_offset_rs * 2;
            load_offset_pos_0 = valid_offset_0 ? load_offset_pos_0 : -1;

            // 加载 npq = 0 对应的offset值
            offset_val0.i32x2_data = __builtin_amdgcn_buffer_load_dword(
                offset_desc.i32x4_data, load_offset_pos_0, 0, 0, 0);

            int mapped_in_n0 = mapped_n0;
            int mapped_in_h0 = mapped_p0 * stride_h + r_index * dilation_h - pad_h;
            int mapped_in_w0 = mapped_q0 * stride_w + s_index * dilation_w - pad_w;

            float float_im_h0 = mapped_in_h0 + static_cast<float>(offset_val0.fp32x2_data[0]);
            float float_im_w0 = mapped_in_w0 + static_cast<float>(offset_val0.fp32x2_data[1]);

            fp32x16 temp_res;

            fp32x16 weight_coord;
            igemm_dmcn_get_coordinate_weight_nhwc(weight_coord,
                                                  float_im_h0,
                                                  float_im_w0,
                                                  batch_size,
                                                  height,
                                                  width,
                                                  channels,
                                                  mapped_c0,
                                                  mapped_in_n0,
                                                  data_im,
                                                  bp_dir);
            temp_res[0] = res0[0] * weight_coord[0];
            temp_res[1] = res0[1] * weight_coord[1];
            temp_res[2] = res0[2] * weight_coord[2];
            temp_res[3] = res0[3] * weight_coord[3];

            temp_res[4] = res1[0] * weight_coord[4];
            temp_res[5] = res1[1] * weight_coord[5];
            temp_res[6] = res1[2] * weight_coord[6];
            temp_res[7] = res1[3] * weight_coord[7];

            temp_res[8]  = res2[0] * weight_coord[8];
            temp_res[9]  = res2[1] * weight_coord[9];
            temp_res[10] = res2[2] * weight_coord[10];
            temp_res[11] = res2[3] * weight_coord[11];

            temp_res[12] = res3[0] * weight_coord[12];
            temp_res[13] = res3[1] * weight_coord[13];
            temp_res[14] = res3[2] * weight_coord[14];
            temp_res[15] = res3[3] * weight_coord[15];

            float thread_sum_0 = 0.0f;
            thread_sum_0 += temp_res[0];
            thread_sum_0 += temp_res[1];
            thread_sum_0 += temp_res[2];
            thread_sum_0 += temp_res[3];

            thread_sum_0 += temp_res[4];
            thread_sum_0 += temp_res[5];
            thread_sum_0 += temp_res[6];
            thread_sum_0 += temp_res[7];

            thread_sum_0 += temp_res[8];
            thread_sum_0 += temp_res[9];
            thread_sum_0 += temp_res[10];
            thread_sum_0 += temp_res[11];

            thread_sum_0 += temp_res[12];
            thread_sum_0 += temp_res[13];
            thread_sum_0 += temp_res[14];
            thread_sum_0 += temp_res[15];

            thread_sum_0 = sum_fp32_cross_lane_stride16(thread_sum_0, bpermute_addr);

            int store_npq   = blk_map_out_npq + (wave_id & 3) * 16 + (lane_id & 15);
            int store_rs    = r_index * kernel_w + s_index;
            int valid_store = LESS_U32(lane_id, 16) &&
                              LESS_U32(load_offset_npq_0, batch_size * out_h * out_w) &&
                              LESS_U32(store_rs, kernel_h * kernel_w);
            int grad_offset_store_pos =
                store_npq * (kernel_h * kernel_w * 2) + store_rs * 2 + bp_dir;
            grad_offset_store_pos = valid_store ? grad_offset_store_pos : -1;

            // __builtin_amdgcn_buffer_store_dword(__builtin_bit_cast(int32_t, thread_sum_0),
            // grad_offset_desc.i32x4_data, grad_offset_store_pos, 0, 0, 0);

            __builtin_amdgcn_buffer_atomic_fadd_f32(
                thread_sum_0, grad_offset_desc.i32x4_data, grad_offset_store_pos, 0, 0);

            bp_dir = 1;
            igemm_dmcn_get_coordinate_weight_nhwc(weight_coord,
                                                  float_im_h0,
                                                  float_im_w0,
                                                  batch_size,
                                                  height,
                                                  width,
                                                  channels,
                                                  mapped_c0,
                                                  mapped_in_n0,
                                                  data_im,
                                                  bp_dir);
            temp_res[0] = res0[0] * weight_coord[0];
            temp_res[1] = res0[1] * weight_coord[1];
            temp_res[2] = res0[2] * weight_coord[2];
            temp_res[3] = res0[3] * weight_coord[3];

            temp_res[4] = res1[0] * weight_coord[4];
            temp_res[5] = res1[1] * weight_coord[5];
            temp_res[6] = res1[2] * weight_coord[6];
            temp_res[7] = res1[3] * weight_coord[7];

            temp_res[8]  = res2[0] * weight_coord[8];
            temp_res[9]  = res2[1] * weight_coord[9];
            temp_res[10] = res2[2] * weight_coord[10];
            temp_res[11] = res2[3] * weight_coord[11];

            temp_res[12] = res3[0] * weight_coord[12];
            temp_res[13] = res3[1] * weight_coord[13];
            temp_res[14] = res3[2] * weight_coord[14];
            temp_res[15] = res3[3] * weight_coord[15];

            thread_sum_0 = 0.0f;
            thread_sum_0 += temp_res[0];
            thread_sum_0 += temp_res[1];
            thread_sum_0 += temp_res[2];
            thread_sum_0 += temp_res[3];

            thread_sum_0 += temp_res[4];
            thread_sum_0 += temp_res[5];
            thread_sum_0 += temp_res[6];
            thread_sum_0 += temp_res[7];

            thread_sum_0 += temp_res[8];
            thread_sum_0 += temp_res[9];
            thread_sum_0 += temp_res[10];
            thread_sum_0 += temp_res[11];

            thread_sum_0 += temp_res[12];
            thread_sum_0 += temp_res[13];
            thread_sum_0 += temp_res[14];
            thread_sum_0 += temp_res[15];

            thread_sum_0 = sum_fp32_cross_lane_stride16(thread_sum_0, bpermute_addr);

            grad_offset_store_pos = valid_store ? grad_offset_store_pos + 1 : -1;

            __builtin_amdgcn_buffer_atomic_fadd_f32(
                thread_sum_0, grad_offset_desc.i32x4_data, grad_offset_store_pos, 0, 0);
        }
    }
}
#define CVT_NUM_THREAD 8

__global__ void
float2float16(_Float16* dst, float* src, size_t len, int blockNum, int reminderInblock)
{

    int threadidInGrid = threadIdx.x + blockIdx.x * blockDim.x;

    int offsetRead = threadidInGrid * CVT_NUM_THREAD;

    if(blockIdx.x == (blockNum - 1) && reminderInblock != 0)
    {
        if(offsetRead + 0 < len)
        {
            dst[offsetRead + 0] = (_Float16)src[offsetRead + 0];
        }

        if(offsetRead + 1 < len)
        {
            dst[offsetRead + 1] = (_Float16)src[offsetRead + 1];
        }

        if(offsetRead + 2 < len)
        {
            dst[offsetRead + 2] = (_Float16)src[offsetRead + 2];
        }

        if(offsetRead + 3 < len)
        {
            dst[offsetRead + 3] = (_Float16)src[offsetRead + 3];
        }

        if(offsetRead + 4 < len)
        {
            dst[offsetRead + 4] = (_Float16)src[offsetRead + 4];
        }

        if(offsetRead + 5 < len)
        {
            dst[offsetRead + 5] = (_Float16)src[offsetRead + 5];
        }

        if(offsetRead + 6 < len)
        {
            dst[offsetRead + 6] = (_Float16)src[offsetRead + 6];
        }

        if(offsetRead + 7 < len)
        {
            dst[offsetRead + 7] = (_Float16)src[offsetRead + 7];
        }
    }
    else
    {
        dst[offsetRead + 0] = (_Float16)src[offsetRead + 0];
        dst[offsetRead + 1] = (_Float16)src[offsetRead + 1];
        dst[offsetRead + 2] = (_Float16)src[offsetRead + 2];
        dst[offsetRead + 3] = (_Float16)src[offsetRead + 3];
        dst[offsetRead + 4] = (_Float16)src[offsetRead + 4];
        dst[offsetRead + 5] = (_Float16)src[offsetRead + 5];
        dst[offsetRead + 6] = (_Float16)src[offsetRead + 6];
        dst[offsetRead + 7] = (_Float16)src[offsetRead + 7];
    }
}

template <typename T>
void modulated_deformable_col2im_coord_cpu_kernel_fused_nhwc(const T* grad_output,
                                                             const T* weight,
                                                             const T* data_im,
                                                             const T* data_offset,
                                                             const T* data_mask,
                                                             const int channels,
                                                             const int height,
                                                             const int width,
                                                             const int kernel_h,
                                                             const int kernel_w,
                                                             const int pad_h,
                                                             const int pad_w,
                                                             const int stride_h,
                                                             const int stride_w,
                                                             const int dilation_h,
                                                             const int dilation_w,
                                                             const int group,
                                                             const int deformable_group,
                                                             const int batch_size,
                                                             const int height_col,
                                                             const int width_col,
                                                             const int out_channels,
                                                             T* grad_offset,
                                                             T* grad_mask)
{
    const int channels_per_group        = channels / group;
    const int out_channels_per_group    = out_channels / group;
    const int channels_per_deform_group = channels / deformable_group;

    // 初始化梯度为0
    const int grad_offset_size =
        batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
    const int grad_mask_size =
        batch_size * height_col * width_col * kernel_h * kernel_w * deformable_group;

    for(int i = 0; i < grad_offset_size; ++i)
    {
        grad_offset[i] = 0;
    }
    for(int i = 0; i < grad_mask_size; ++i)
    {
        grad_mask[i] = 0;
    }

    for(int b = 0; b < batch_size; ++b)
    {
        const T* data_im_ptr = data_im + b * height * width * channels;
        const T* data_offset_ptr =
            data_offset + b * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
        const T* data_mask_ptr =
            data_mask + b * height_col * width_col * kernel_h * kernel_w * deformable_group;

        for(int h = 0; h < height_col; ++h)
        {
            for(int w = 0; w < width_col; ++w)
            {
                const int w_in = w * stride_w - pad_w;
                const int h_in = h * stride_h - pad_h;

                const int spatial_index = h * width_col + w;

                for(int dg = 0; dg < deformable_group; ++dg)
                {
                    for(int i = 0; i < kernel_h; ++i)
                    {
                        for(int j = 0; j < kernel_w; ++j)
                        {
                            const int kernel_index = i * kernel_w + j;

                            // NHWC布局下的offset和mask索引
                            const int data_offset_h_ptr =
                                (spatial_index * 2 * kernel_h * kernel_w * deformable_group +
                                 dg * 2 * kernel_h * kernel_w + 2 * kernel_index);
                            const int data_offset_w_ptr = data_offset_h_ptr + 1;
                            const int data_mask_hw_ptr =
                                (spatial_index * kernel_h * kernel_w * deformable_group +
                                 dg * kernel_h * kernel_w + kernel_index);

                            const T offset_h = data_offset_ptr[data_offset_h_ptr];
                            const T offset_w = data_offset_ptr[data_offset_w_ptr];
                            const T mask     = data_mask_ptr[data_mask_hw_ptr];

                            // 计算采样位置
                            T inv_h = h_in + i * dilation_h + offset_h;
                            T inv_w = w_in + j * dilation_w + offset_w;

                            // 使用double进行累加以提高精度
                            double val_x = 0, val_y = 0, mval = 0;

                            for(int channel_index = 0; channel_index < channels_per_deform_group;
                                ++channel_index)
                            {
                                const int global_input_channel =
                                    dg * channels_per_deform_group + channel_index;
                                const int group_idx = global_input_channel / channels_per_group;
                                const int input_channel_in_group =
                                    global_input_channel % channels_per_group;

                                // 计算data_col_val
                                double data_col_val = 0;
                                for(int k = 0; k < out_channels_per_group; k++)
                                {
                                    const int global_output_channel =
                                        group_idx * out_channels_per_group + k;
                                    const int grad_output_idx =
                                        ((b * height_col * width_col + spatial_index) *
                                             out_channels +
                                         global_output_channel);
                                    const int weight_idx =
                                        ((global_output_channel * kernel_h * kernel_w +
                                          i * kernel_w + j) *
                                             channels_per_group +
                                         input_channel_in_group);
                                    // const int weight_idx = (((group_idx * out_channels_per_group
                                    // + k) * channels_per_group +
                                    //     input_channel_in_group) * kernel_h + i) * kernel_w + j;
                                    data_col_val +=
                                        static_cast<double>(weight[weight_idx]) *
                                        static_cast<double>(grad_output[grad_output_idx]);
                                }

                                if(inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
                                {
                                    inv_h = inv_w = -2;
                                }
                                else
                                {
                                    mval += data_col_val *
                                            dmcn_im2col_bilinear_nhwc(data_im_ptr,
                                                                      height,
                                                                      width,
                                                                      channels,
                                                                      inv_h,
                                                                      inv_w,
                                                                      global_input_channel);
                                }

                                // 计算坐标权重
                                const T weight_coord_x =
                                    dmcn_get_coordinate_weight_nhwc(inv_h,
                                                                    inv_w,
                                                                    height,
                                                                    width,
                                                                    channels,
                                                                    data_im_ptr,
                                                                    global_input_channel,
                                                                    0);
                                const T weight_coord_y =
                                    dmcn_get_coordinate_weight_nhwc(inv_h,
                                                                    inv_w,
                                                                    height,
                                                                    width,
                                                                    channels,
                                                                    data_im_ptr,
                                                                    global_input_channel,
                                                                    1);

                                val_x += static_cast<double>(weight_coord_x) * data_col_val;
                                val_y += static_cast<double>(weight_coord_y) * data_col_val;
                            }

                            // 存储梯度
                            const int offset_idx_x =
                                ((b * height_col * width_col + spatial_index) * 2 * kernel_h *
                                     kernel_w * deformable_group +
                                 dg * 2 * kernel_h * kernel_w + 2 * kernel_index);
                            const int offset_idx_y = offset_idx_x + 1;
                            const int mask_idx     = ((b * height_col * width_col + spatial_index) *
                                                      kernel_h * kernel_w * deformable_group +
                                                  dg * kernel_h * kernel_w + kernel_index);

                            grad_offset[offset_idx_x] = static_cast<T>(val_x);
                            grad_offset[offset_idx_y] = static_cast<T>(val_y);
                            grad_mask[mask_idx]       = static_cast<T>(mval);
                        }
                    }
                }
            }
        }
    }
}

// 比较结果
template <typename T>
bool compare_results(
    const T* cpu_result, const T* gpu_result, int size, const char* name, T tolerance = 1e-4)
{
    bool match               = true;
    int mismatch_count       = 0;
    const int max_mismatches = 10;

    for(int i = 0; i < size; ++i)
    {
        T diff          = std::abs(cpu_result[i] - gpu_result[i]);
        T max_val       = std::max(std::abs(cpu_result[i]), std::abs(gpu_result[i]));
        T relative_diff = (max_val > 1e-6) ? diff / max_val : diff;

        if(diff > tolerance && relative_diff > tolerance)
        {
            if(mismatch_count < max_mismatches)
            {
                std::cout << "Mismatch in " << name << " at index " << i
                          << ": CPU=" << cpu_result[i] << ", GPU=" << gpu_result[i]
                          << ", diff=" << diff << ", relative=" << relative_diff << std::endl;
            }
            mismatch_count++;
            match = false;
        }
    }

    if(mismatch_count > 0)
    {
        std::cout << "Total mismatches in " << name << ": " << mismatch_count << " out of " << size
                  << std::endl;
    }

    return match;
}

// 预热GPU
void warmup_gpu()
{
    float* d_temp;
    HIP_CHECK(hipMalloc(&d_temp, sizeof(float)));
    HIP_CHECK(hipFree(d_temp));
    hipDeviceSynchronize();
}

#define COLOR_RED "\033[1;31m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_RESET "\033[0m"

int main()
{
    // 预热GPU
    warmup_gpu();

    // 基础参数
    int batch_size, channels, height, width, kernel_h, kernel_w;
    int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
    int deformable_group, group, out_channels;

    // 设置固定参数
    batch_size       = 6;
    channels         = 256;
    height           = 58;
    width            = 100;
    kernel_h         = 3;
    kernel_w         = 3;
    pad_h            = 1;
    pad_w            = 1;
    stride_h         = 1;
    stride_w         = 1;
    dilation_h       = 1;
    dilation_w       = 1;
    deformable_group = 1;
    group            = 1;
    out_channels     = 256;

    // 验证参数
    if((channels % group) != 0)
    {
        std::cout << COLOR_RED << "PARAM ERROR: channels (" << channels
                  << ") must be divisible by group (" << group << ") " << COLOR_RESET << std::endl;
        return 1;
    }

    if((channels % deformable_group) != 0)
    {
        std::cout << COLOR_RED << "PARAM ERROR: channels (" << channels
                  << ") must be divisible by deformable_group (" << deformable_group << ") "
                  << COLOR_RESET << std::endl;
        return 1;
    }

    // 计算输出尺寸
    const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_col  = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // 计算各种张量大小 - NHWC布局
    const int data_im_size = batch_size * height * width * channels;
    const int data_offset_size =
        batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
    const int data_mask_size =
        batch_size * height_col * width_col * kernel_h * kernel_w * deformable_group;
    const int grad_output_size = batch_size * height_col * width_col * out_channels;
    const int weight_size = out_channels * kernel_h * kernel_w * (channels / group); // NHWC权重布局

    std::cout << "=== DCNv2 Coordinate Gradient ===" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  channels: " << channels << std::endl;
    std::cout << "  out_channels: " << out_channels << std::endl;
    std::cout << "  height: " << height << ", width: " << width << std::endl;
    std::cout << "  kernel: " << kernel_h << "x" << kernel_w << std::endl;
    std::cout << "  group: " << group << std::endl;
    std::cout << "  deformable_group: " << deformable_group << std::endl;
    std::cout << "  height_col: " << height_col << ", width_col: " << width_col << std::endl;
    std::cout << "  GPU total elements: " << n << std::endl;

    if(channels % 64)
    {
        std::cout << "channels must be multiply of " << C_PER_BLK << std::endl;
    }

    // 分配主机内存
    std::vector<DATA_TYPE> h_grad_output(grad_output_size);
    std::vector<DATA_TYPE> h_weight(weight_size);
    std::vector<DATA_TYPE> h_data_im(data_im_size);
    std::vector<DATA_TYPE> h_data_offset(data_offset_size);
    std::vector<DATA_TYPE> h_data_mask(data_mask_size);
    std::vector<DATA_TYPE> h_grad_offset_cpu(data_offset_size);
    std::vector<DATA_TYPE> h_grad_offset_gpu(data_offset_size);
    std::vector<DATA_TYPE> h_grad_mask_cpu(data_mask_size);
    std::vector<DATA_TYPE> h_grad_mask_gpu(data_mask_size);

    // 初始化随机数据
    float bottom = -3.0f;
    float top    = 3.0f;
    for(size_t i = 0; i < h_data_im.size(); i++)
    {
        h_data_im.data()[i] = static_cast<_Float16>((RAN_GEN(bottom, top)));
    }

    for(size_t i = 0; i < h_data_offset.size(); i++)
    {
        h_data_offset.data()[i] = static_cast<_Float16>((RAN_GEN(-0.05f, 0.05f)));
    }

    for(size_t i = 0; i < h_data_mask.size(); i++)
    {
        h_data_mask.data()[i] = static_cast<_Float16>(RAN_GEN(0.0f, 1.0f));
    }

    for(size_t i = 0; i < h_weight.size(); i++)
    {
        h_weight.data()[i] = static_cast<_Float16>(RAN_GEN(-0.1f, 0.1f));
    }

    for(size_t i = 0; i < h_grad_output.size(); i++)
    {
        h_grad_output.data()[i] = static_cast<_Float16>(RAN_GEN(-0.1f, 0.1f));
    }

    std::cout << "数据初始化完成:" << std::endl;
    std::cout << "  h_data_im: " << h_data_im.size() << " elements" << std::endl;
    std::cout << "  h_data_offset: " << h_data_offset.size() << " elements" << std::endl;
    std::cout << "  h_data_mask: " << h_data_mask.size() << " elements" << std::endl;
    std::cout << "  h_weight: " << h_weight.size() << " elements" << std::endl;
    std::cout << "  h_grad_output: " << h_grad_output.size() << " elements" << std::endl;

    // 分配设备内存
    DATA_TYPE *d_grad_output, *d_weight, *d_data_im, *d_data_offset, *d_data_mask;
    DATA_TYPE *d_grad_offset, *d_grad_float2fp16;

    HIP_CHECK(hipMalloc(&d_grad_output, grad_output_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_data_im, data_im_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_data_offset, data_offset_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_data_mask, data_mask_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_grad_offset, data_offset_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_grad_float2fp16, data_offset_size * sizeof(DATA_TYPE)));

    // 初始化设备梯度内存为0
    HIP_CHECK(hipMemset(d_grad_offset, 0, data_offset_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMemset(d_grad_float2fp16, 0, data_offset_size * sizeof(DATA_TYPE)));

    // 拷贝输入数据到设备
    HIP_CHECK(hipMemcpy(d_grad_output,
                        h_grad_output.data(),
                        grad_output_size * sizeof(DATA_TYPE),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_weight, h_weight.data(), weight_size * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_data_im, h_data_im.data(), data_im_size * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_data_offset,
                        h_data_offset.data(),
                        data_offset_size * sizeof(DATA_TYPE),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_data_mask,
                        h_data_mask.data(),
                        data_mask_size * sizeof(DATA_TYPE),
                        hipMemcpyHostToDevice));
#if VERIFY
    // 运行CPU
    modulated_deformable_col2im_coord_cpu_kernel_fused_nhwc(h_grad_output.data(),
                                                            h_weight.data(),
                                                            h_data_im.data(),
                                                            h_data_offset.data(),
                                                            h_data_mask.data(),
                                                            channels,
                                                            height,
                                                            width,
                                                            kernel_h,
                                                            kernel_w,
                                                            pad_h,
                                                            pad_w,
                                                            stride_h,
                                                            stride_w,
                                                            dilation_h,
                                                            dilation_w,
                                                            group,
                                                            deformable_group,
                                                            batch_size,
                                                            height_col,
                                                            width_col,
                                                            out_channels,
                                                            h_grad_offset_cpu.data(),
                                                            h_grad_mask_cpu.data());
#endif

    // 运行GPU
    dim3 igemm_grid(1, 1, 1);
    dim3 igemm_block(256, 1, 1);

    igemm_grid.x = DIV_CEIL(channels, C_PER_BLK);
    igemm_grid.y = DIV_CEIL(batch_size * height_col * width_col, NPQ_PER_BLK);
    igemm_grid.z = 1;

    OffsetIgemmParam igemm_param;
    memset(&igemm_param, 0, sizeof(igemm_param));
    igemm_param.grad_output = (void*)d_grad_output;
    igemm_param.weight      = (void*)d_weight;
    igemm_param.data_im     = (void*)d_data_im;
    igemm_param.data_offset = (void*)d_data_offset;
    igemm_param.data_mask   = (void*)d_data_mask;
    igemm_param.grad_offset = (void*)d_grad_offset;
    igemm_param.grad_mask   = (void*)d_grad_mask;

    igemm_param.batch_size       = batch_size;
    igemm_param.channels         = channels;
    igemm_param.height           = height;
    igemm_param.width            = width;
    igemm_param.out_channels     = out_channels;
    igemm_param.kernel_h         = kernel_h;
    igemm_param.kernel_w         = kernel_w;
    igemm_param.pad_h            = pad_h;
    igemm_param.pad_w            = pad_w;
    igemm_param.stride_h         = stride_h;
    igemm_param.stride_w         = stride_w;
    igemm_param.dilation_h       = dilation_h;
    igemm_param.dilation_w       = dilation_w;
    igemm_param.group            = group;
    igemm_param.deformable_group = deformable_group;

    igemm_param.out_h = height_col;
    igemm_param.out_w = width_col;

    int blockNum = (data_offset_size - 1) / (256 * CVT_NUM_THREAD) + 1;
    int reminder = data_offset_size % (256 * CVT_NUM_THREAD);
    dim3 threadsCvt(256, 1, 1);
    dim3 groupsCvt(blockNum, 1, 1);

    // 预热运行
    hipLaunchKernelGGL(
        igemm_offset_fp32_nhwc, dim3(igemm_grid), dim3(igemm_block), 0, 0, igemm_param);

    float2float16<<<groupsCvt, threadsCvt, 0, 0>>>(
        d_grad_float2fp16.data(), d_grad_offset, data_offset_size, blockNum, reminder);

    HIP_CHECK(hipDeviceSynchronize());

    // 拷贝结果回主机
    HIP_CHECK(hipMemcpy(h_grad_offset_gpu.data(),
                        d_grad_float2fp16,
                        data_offset_size * sizeof(DATA_TYPE),
                        hipMemcpyDeviceToHost));

#if VERIFY
    bool offset_match = compare_results(h_grad_offset_cpu.data(),
                                        h_grad_offset_gpu.data(),
                                        std::min(1000, data_offset_size),
                                        "grad_offset",
                                        1e-3f);

    if(offset_match /* && mask_match*/)
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

    std::cout << "------------------------------ HOST GRAD_OFFSET RES ----------------------------"
              << std::endl;
    print_matrix<DATA_TYPE>(&h_grad_offset_cpu[0], h_grad_offset_cpu.size(), 0, 128);

    std::cout
        << "------------------------------ DEVICE GRAD_OFFSET RES ----------------------------"
        << std::endl;
    print_matrix<DATA_TYPE>(&h_grad_offset_gpu[0], h_grad_offset_gpu.size(), 0, 128);

#endif

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float cur_time   = 0.0f;
    float total_time = 0.0f;

    for(int iter = 0; iter < PERFORM_MESURE_TIMES; iter++)
    {
        hipEventRecord(start, 0);
        hipLaunchKernelGGL(
            igemm_offset_fp32_nhwc, dim3(igemm_grid), dim3(igemm_block), 0, 0, igemm_param);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&cur_time, start, stop);
        total_time += cur_time;
    }

    printf("kernel time(ms): %7.5f\n", total_time / PERFORM_MESURE_TIMES);

    // 清理设备内存
    HIP_CHECK(hipFree(d_grad_output));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_data_im));
    HIP_CHECK(hipFree(d_data_offset));
    HIP_CHECK(hipFree(d_data_mask));
    HIP_CHECK(hipFree(d_grad_offset));
    HIP_CHECK(hipFree(d_grad_mask));

    return 0;
}
