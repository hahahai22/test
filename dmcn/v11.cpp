#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <fstream>

typedef _Float16 DATA_TYPE;

// 文件保存函数
void save_tensor_to_file(const std::vector<DATA_TYPE>& data,
                         const std::vector<int>& shape,
                         const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if(!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    // 保存数据
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(DATA_TYPE));
    file.close();

    std::cout << "Saved " << data.size() << " elements to " << filename << " with shape ["
              << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "]"
              << std::endl;
}

// 新的文件加载函数
std::vector<DATA_TYPE> load_tensor_from_file(const std::string& filename,
                                             const std::vector<int>& expected_shape)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if(!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    // 获取文件大小
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 计算期望的数据大小
    int total_size = 1;
    for(int dim : expected_shape)
    {
        total_size *= dim;
    }
    size_t expected_bytes = total_size * sizeof(DATA_TYPE);

    if(file_size != expected_bytes)
    {
        std::cerr << "File size mismatch for " << filename << ": expected " << expected_bytes
                  << " bytes, got " << file_size << " bytes" << std::endl;
        return {};
    }

    std::vector<DATA_TYPE> data(total_size);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    std::cout << "Loaded " << total_size << " elements from " << filename
              << " with expected shape [" << expected_shape[0] << ", " << expected_shape[1] << ", "
              << expected_shape[2] << ", " << expected_shape[3] << "]" << std::endl;

    return data;
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

// 双线性插值函数 - NHWC版本
template <typename T>
__device__ __host__ T dmcn_im2col_bilinear_nhwc(
    const T* input, const int height, const int width, const int channels, T h, T w, int c)
{
    if(h <= -1 || w <= -1 || h >= height || w >= width)
        return 0;

    int h_low  = floorf(h);
    int w_low  = floorf(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    T lh = h - h_low;
    T lw = w - w_low;
    T hh = 1 - lh, hw = 1 - lw;

    // NHWC索引计算: (h * width + w) * channels + c
    T v1 = 0;
    if(h_low >= 0 && w_low >= 0)
        v1 = input[(h_low * width + w_low) * channels + c];
    T v2 = 0;
    if(h_low >= 0 && w_high <= width - 1)
        v2 = input[(h_low * width + w_high) * channels + c];
    T v3 = 0;
    if(h_high <= height - 1 && w_low >= 0)
        v3 = input[(h_high * width + w_low) * channels + c];
    T v4 = 0;
    if(h_high <= height - 1 && w_high <= width - 1)
        v4 = input[(h_high * width + w_high) * channels + c];

    T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <typename T>
__device__ __host__ T dmcn_get_coordinate_weight_nhwc(T argmax_h,
                                                      T argmax_w,
                                                      const int height,
                                                      const int width,
                                                      const int channels,
                                                      const T* im_data,
                                                      int c,
                                                      const int bp_dir)
{
    if(argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
    {
        return 0;
    }

    int argmax_h_low  = floorf(argmax_h);
    int argmax_w_low  = floorf(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    T weight = 0;

    // NHWC索引计算: (h * width + w) * channels + c
    if(bp_dir == 0) // 计算x方向的梯度（即对Δx的梯度）
    {
        if(argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_w_low + 1 - argmax_w) *
                      im_data[(argmax_h_low * width + argmax_w_low) * channels + c];
        if(argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += -1 * (argmax_w - argmax_w_low) *
                      im_data[(argmax_h_low * width + argmax_w_high) * channels + c];
        if(argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += (argmax_w_low + 1 - argmax_w) *
                      im_data[(argmax_h_high * width + argmax_w_low) * channels + c];
        if(argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_w - argmax_w_low) *
                      im_data[(argmax_h_high * width + argmax_w_high) * channels + c];
    }
    else if(bp_dir == 1) // 计算y方向的梯度（即对Δy的梯度）
    {
        if(argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_h_low + 1 - argmax_h) *
                      im_data[(argmax_h_low * width + argmax_w_low) * channels + c];
        if(argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += (argmax_h_low + 1 - argmax_h) *
                      im_data[(argmax_h_low * width + argmax_w_high) * channels + c];
        if(argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += -1 * (argmax_h - argmax_h_low) *
                      im_data[(argmax_h_high * width + argmax_w_low) * channels + c];
        if(argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_h - argmax_h_low) *
                      im_data[(argmax_h_high * width + argmax_w_high) * channels + c];
    }

    return weight;
}

template <typename T>
__global__ void modulated_deformable_col2im_coord_gpu_kernel_nhwc(const int n,
                                                                  const T* grad_output,
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
    HIP_1D_KERNEL_LOOP(index, n)
    {
        T val = 0, mval = 0;
        int w = index % width_col;
        int h = (index / width_col) % height_col;

        // offset_channels = 2 * kernel_h * kernel_w * deformable_group
        int c = (index / width_col / height_col) % (2 * kernel_h * kernel_w * deformable_group);
        int b = (index / width_col / height_col) / (2 * kernel_h * kernel_w * deformable_group);

        const int deformable_group_index = c / (2 * kernel_h * kernel_w);
        const int col_step               = kernel_h * kernel_w;

        // 计算每个group的通道数
        const int channels_per_group        = channels / group;
        const int out_channels_per_group    = out_channels / group;
        const int channels_per_deform_group = channels / deformable_group;

        // NHWC布局: data_im的shape为[batch, height, width, channels]
        const T* data_im_ptr = data_im + b * height * width * channels; // 整个batch的起始位置

        const T* data_offset_ptr =
            data_offset + b * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
        const T* data_mask_ptr =
            data_mask + b * height_col * width_col * kernel_h * kernel_w * deformable_group;

        const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

        // 遍历该deformable group内的所有位置
        for(int col_c = (offset_c / 2); col_c < channels_per_deform_group * kernel_h * kernel_w;
            col_c += col_step)
        {
            const int bp_dir = offset_c % 2;

            // 解析卷积核位置和输入通道
            int j = col_c % kernel_w;
            int i = (col_c / kernel_w) % kernel_h;
            int input_channel_in_deform =
                col_c / (kernel_h * kernel_w); // 在当前deformable group内的输入通道索引

            // 计算输入坐标
            int w_in = w * stride_w - pad_w;
            int h_in = h * stride_h - pad_h;

            // NHWC布局下的offset和mask索引
            const int data_offset_h_ptr =
                (((h * width_col + w) * 2 * kernel_h * kernel_w * deformable_group +
                  deformable_group_index * 2 * kernel_h * kernel_w + 2 * (i * kernel_w + j)));
            const int data_offset_w_ptr = data_offset_h_ptr + 1;
            const int data_mask_hw_ptr =
                ((h * width_col + w) * kernel_h * kernel_w * deformable_group +
                 deformable_group_index * kernel_h * kernel_w + (i * kernel_w + j));

            const T offset_h = data_offset_ptr[data_offset_h_ptr];
            const T offset_w = data_offset_ptr[data_offset_w_ptr];
            const T mask     = data_mask_ptr[data_mask_hw_ptr];

            // 计算采样位置
            T inv_h = h_in + i * dilation_h + offset_h;
            T inv_w = w_in + j * dilation_w + offset_w;

            // 计算 data_col 元素
            T data_col_val = 0;

            // 计算全局输入通道
            int global_input_channel =
                deformable_group_index * channels_per_deform_group + input_channel_in_deform;
            int group_index            = global_input_channel / channels_per_group;
            int input_channel_in_group = global_input_channel % channels_per_group;

            for(int k = 0; k < out_channels_per_group; k++)
            {
                // NHWC布局下的grad_output索引: [b, h, w, group_index * out_channels_per_group + k]
                int global_output_channel = group_index * out_channels_per_group + k;
                int grad_output_idx =
                    ((b * height_col * width_col + h * width_col + w) * out_channels +
                     global_output_channel);

                // NHWC布局下的weight索引: weight形状为[out_channels, kernel_h, kernel_w,
                // channels/group]
                int weight_idx = ((global_output_channel * kernel_h * kernel_w + i * kernel_w + j) *
                                      channels_per_group +
                                  input_channel_in_group);

                data_col_val += weight[weight_idx] * grad_output[grad_output_idx];
            }

            if(inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
            {
                inv_h = inv_w = -2;
            }

            const T weight_coord = dmcn_get_coordinate_weight_nhwc<DATA_TYPE>(
                inv_h, inv_w, height, width, channels, data_im_ptr, global_input_channel, bp_dir);

            val += weight_coord * data_col_val * mask;
        }

        // NHWC布局下的grad_offset索引
        int grad_offset_idx = ((b * height_col * width_col + h * width_col + w) * 2 * kernel_h *
                                   kernel_w * deformable_group +
                               c);
        grad_offset[grad_offset_idx] = val;
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
    for(int b = 0; b < batch_size; ++b)
    {
        for(int h = 0; h < height_col; ++h)
        {
            for(int w = 0; w < width_col; ++w)
            {
                for(int dg = 0; dg < deformable_group; ++dg)
                {
                    for(int c = 0; c < 2 * kernel_h * kernel_w; ++c)
                    {
                        int idx = ((b * height_col * width_col + h * width_col + w) * 2 * kernel_h *
                                       kernel_w * deformable_group +
                                   dg * 2 * kernel_h * kernel_w + c);
                        grad_offset[idx] = 0;
                    }

                    for(int k = 0; k < kernel_h * kernel_w; ++k)
                    {
                        int idx = ((b * height_col * width_col + h * width_col + w) * kernel_h *
                                       kernel_w * deformable_group +
                                   dg * kernel_h * kernel_w + k);
                        grad_mask[idx] = 0;
                    }
                }
            }
        }
    }

    for(int b = 0; b < batch_size; ++b)
    {
        // NHWC布局指针
        const T* data_im_ptr = data_im + b * height * width * channels;
        const T* data_offset_ptr =
            data_offset + b * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
        const T* data_mask_ptr =
            data_mask + b * height_col * width_col * kernel_h * kernel_w * deformable_group;

        for(int h = 0; h < height_col; ++h)
        {
            for(int w = 0; w < width_col; ++w)
            {
                // 计算输入坐标
                int w_in = w * stride_w - pad_w;
                int h_in = h * stride_h - pad_h;

                for(int dg = 0; dg < deformable_group; ++dg)
                {
                    // 拆开循环：先处理每个核位置的偏移计算
                    for(int i = 0; i < kernel_h; ++i)
                    {
                        for(int j = 0; j < kernel_w; ++j)
                        {
                            T val_x = 0, val_y = 0, mval = 0;

                            // NHWC布局下的offset和mask索引
                            const int data_offset_h_ptr =
                                ((h * width_col + w) * 2 * kernel_h * kernel_w * deformable_group +
                                 dg * 2 * kernel_h * kernel_w + 2 * (i * kernel_w + j));
                            const int data_offset_w_ptr = data_offset_h_ptr + 1;
                            const int data_mask_hw_ptr =
                                ((h * width_col + w) * kernel_h * kernel_w * deformable_group +
                                 dg * kernel_h * kernel_w + (i * kernel_w + j));

                            const T offset_h = data_offset_ptr[data_offset_h_ptr];
                            const T offset_w = data_offset_ptr[data_offset_w_ptr];
                            const T mask     = data_mask_ptr[data_mask_hw_ptr];

                            // 计算采样位置
                            T inv_h = h_in + i * dilation_h + offset_h;
                            T inv_w = w_in + j * dilation_w + offset_w;

                            // 对于每个核位置，循环所有通道
                            for(int channel_index = 0; channel_index < channels_per_deform_group;
                                ++channel_index)
                            {
                                // 计算 data_col 元素
                                T data_col_val = 0;

                                // 从deformable_group的channel推导出当前channel位于哪个group中
                                int global_input_channel =
                                    dg * channels_per_deform_group + channel_index;
                                int group_idx = global_input_channel / channels_per_group;
                                int input_channel_in_group =
                                    global_input_channel % channels_per_group;

                                for(int k = 0; k < out_channels_per_group; k++)
                                {
                                    int global_output_channel =
                                        group_idx * out_channels_per_group + k;
                                    // NHWC布局下的grad_output索引
                                    int grad_output_idx =
                                        ((b * height_col * width_col + h * width_col + w) *
                                             out_channels +
                                         global_output_channel);

                                    // NHWC布局下的weight索引: [out_channels, kernel_h, kernel_w,
                                    // channels/group]
                                    int weight_idx = ((global_output_channel * kernel_h * kernel_w +
                                                       i * kernel_w + j) *
                                                          channels_per_group +
                                                      input_channel_in_group);

                                    data_col_val +=
                                        weight[weight_idx] * grad_output[grad_output_idx];
                                }

                                // 双线性插值和坐标权重计算
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

                                // 计算x方向的坐标权重
                                const T weight_coord_x =
                                    dmcn_get_coordinate_weight_nhwc(inv_h,
                                                                    inv_w,
                                                                    height,
                                                                    width,
                                                                    channels,
                                                                    data_im_ptr,
                                                                    global_input_channel,
                                                                    0); // bp_dir = 0 for x

                                val_x += weight_coord_x * data_col_val * mask;

                                // 计算y方向的坐标权重
                                const T weight_coord_y =
                                    dmcn_get_coordinate_weight_nhwc(inv_h,
                                                                    inv_w,
                                                                    height,
                                                                    width,
                                                                    channels,
                                                                    data_im_ptr,
                                                                    global_input_channel,
                                                                    1); // bp_dir = 1 for y

                                val_y += weight_coord_y * data_col_val * mask;
                            }

                            // 存储x方向的偏移梯度
                            int offset_idx_x =
                                ((b * height_col * width_col + h * width_col + w) * 2 * kernel_h *
                                     kernel_w * deformable_group +
                                 dg * 2 * kernel_h * kernel_w + 2 * (i * kernel_w + j));
                            grad_offset[offset_idx_x] = val_x;

                            // 存储y方向的偏移梯度
                            int offset_idx_y          = offset_idx_x + 1;
                            grad_offset[offset_idx_y] = val_y;

                            // 存储掩码梯度
                            int mask_idx = ((b * height_col * width_col + h * width_col + w) *
                                                kernel_h * kernel_w * deformable_group +
                                            dg * kernel_h * kernel_w + (i * kernel_w + j));
                            grad_mask[mask_idx] = mval;
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

bool compare_results_fp16(const DATA_TYPE* cpu_result,
                          const DATA_TYPE* gpu_result,
                          int size,
                          const char* name,
                          float tolerance = 1e-4f)
{
    bool match               = true;
    int mismatch_count       = 0;
    const int max_mismatches = 10;

    for(int i = 0; i < size; ++i)
    {
        float cpu_val       = static_cast<float>(cpu_result[i]);
        float gpu_val       = static_cast<float>(gpu_result[i]);
        float diff          = std::abs(cpu_val - gpu_val);
        float max_val       = std::max(std::abs(cpu_val), std::abs(gpu_val));
        float relative_diff = (max_val > 1e-6f) ? diff / max_val : diff;

        if(diff > tolerance && relative_diff > tolerance)
        {
            if(mismatch_count < max_mismatches)
            {
                std::cout << "Mismatch in " << name << " at index " << i << ": CPU=" << cpu_val
                          << ", GPU=" << gpu_val << ", diff=" << diff
                          << ", relative=" << relative_diff << std::endl;
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
    DATA_TYPE* d_temp;
    HIP_CHECK(hipMalloc(&d_temp, sizeof(DATA_TYPE)));
    HIP_CHECK(hipFree(d_temp));
    hipDeviceSynchronize();
}

double RAN_GEN(double A, double B)
{
    double r = (static_cast<double>(rand() / (static_cast<double>(RAND_MAX))) * (B - A)) + A;
    return r;
}

#define COLOR_RED "\033[1;31m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_RESET "\033[0m"

int main()
{
    // 预热GPU
    warmup_gpu();

    const bool VALIDATION_MODE = false;

    // 基础参数
    int batch_size, channels, height, width, kernel_h, kernel_w;
    int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
    int deformable_group, group, out_channels;

    if(VALIDATION_MODE)
    {
        std::ifstream param_file("test_data_nhwc/params.txt");
        if(!param_file)
        {
            std::cerr << "Cannot open params.txt file" << std::endl;
            return 1;
        }

        std::string line;
        while(std::getline(param_file, line))
        {
            size_t pos = line.find('=');
            if(pos != std::string::npos)
            {
                std::string key = line.substr(0, pos);
                int value       = std::stoi(line.substr(pos + 1));

                if(key == "batch_size")
                    batch_size = value;
                else if(key == "channels")
                    channels = value;
                else if(key == "height")
                    height = value;
                else if(key == "width")
                    width = value;
                else if(key == "kernel_h")
                    kernel_h = value;
                else if(key == "kernel_w")
                    kernel_w = value;
                else if(key == "pad_h")
                    pad_h = value;
                else if(key == "pad_w")
                    pad_w = value;
                else if(key == "stride_h")
                    stride_h = value;
                else if(key == "stride_w")
                    stride_w = value;
                else if(key == "dilation_h")
                    dilation_h = value;
                else if(key == "dilation_w")
                    dilation_w = value;
                else if(key == "deformable_group")
                    deformable_group = value;
                else if(key == "group")
                    group = value;
                else if(key == "out_channels")
                    out_channels = value;
            }
        }
        param_file.close();
    }
    else
    {
        // 设置固定参数
        batch_size       = 6;
        channels         = 512;
        height           = 29;
        width            = 50;
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
        out_channels     = 512;
    }

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

    // GPU 核函数的总处理元素
    const int n = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;

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

    // 数据初始化函数
    // auto initialize_random_data = [](std::vector<DATA_TYPE>& data, float min_val, float max_val)
    // {
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<DATA_TYPE> dis(min_val, max_val);

    //     for(size_t i = 0; i < data.size(); ++i)
    //     {
    //         data[i] = static_cast<DATA_TYPE>(dis(gen));
    //     }
    // };

    if(VALIDATION_MODE)
    {
        std::cout << "\nLoading test data (NHWC layout)..." << std::endl;
        // 加载NHWC布局的数据
        h_data_im     = load_tensor_from_file("test_data_nhwc/input.bin",
                                          {batch_size, height, width, channels});
        h_data_offset = load_tensor_from_file(
            "test_data_nhwc/offset.bin",
            {batch_size, height_col, width_col, 2 * kernel_h * kernel_w * deformable_group});
        h_data_mask = load_tensor_from_file(
            "test_data_nhwc/mask.bin",
            {batch_size, height_col, width_col, kernel_h * kernel_w * deformable_group});
        h_weight      = load_tensor_from_file("test_data_nhwc/weight.bin",
                                         {out_channels, kernel_h, kernel_w, channels / group});
        h_grad_output = load_tensor_from_file("test_data_nhwc/grad_output.bin",
                                              {batch_size, height_col, width_col, out_channels});

        // 检查数据是否成功加载
        if(h_data_im.empty() || h_data_offset.empty() || h_data_mask.empty() || h_weight.empty() ||
           h_grad_output.empty())
        {
            std::cerr << "Failed to load test data" << std::endl;
            return 1;
        }
    }
    else
    {
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
    }

    // 分配设备内存
    DATA_TYPE *d_grad_output, *d_weight, *d_data_im, *d_data_offset, *d_data_mask;
    DATA_TYPE *d_grad_offset, *d_grad_mask;

    HIP_CHECK(hipMalloc(&d_grad_output, grad_output_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_data_im, data_im_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_data_offset, data_offset_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_data_mask, data_mask_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_grad_offset, data_offset_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMalloc(&d_grad_mask, data_mask_size * sizeof(DATA_TYPE)));

    // 初始化设备梯度内存为0
    HIP_CHECK(hipMemset(d_grad_offset, 0, data_offset_size * sizeof(DATA_TYPE)));
    HIP_CHECK(hipMemset(d_grad_mask, 0, data_mask_size * sizeof(DATA_TYPE)));

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
#if 0
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
#if 1
    std::cout << "验证CPP的CPU代码" << std::endl;
    std::vector<int> offset_shape = {
        batch_size, height_col, width_col, 2 * kernel_h * kernel_w * deformable_group};
    std::vector<int> mask_shape = {
        batch_size, height_col, width_col, kernel_h * kernel_w * deformable_group};
    if(VALIDATION_MODE)
    {
        save_tensor_to_file(h_grad_offset_cpu, offset_shape, "test_data_nhwc/grad_offset_cpp.bin");
        save_tensor_to_file(h_grad_mask_cpu, mask_shape, "test_data_nhwc/grad_mask_cpp.bin");
    }
#endif

    const int blockDimX = 256;
    const int blockDimY = 1;
    const int blockDimZ = 1;
    const dim3 blockDim(blockDimX, blockDimY, blockDimZ);

    const int gridDimX = 2 * kernel_h * kernel_w * deformable_group;
    const int gridDimY = (height_col * width_col + 64 - 1) / 64;
    const int gridDimZ = batch_size;
    const dim3 gridDim(gridDimX, gridDimY, gridDimZ);

    // 预热运行
    hipLaunchKernelGGL(modulated_deformable_col2im_coord_gpu_kernel_nhwc,
                       gridDim,
                       blockDim,
                       0,
                       0,
                       n,
                       d_grad_output,
                       d_weight,
                       d_data_im,
                       d_data_offset,
                       d_data_mask,
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
                       d_grad_offset,
                       d_grad_mask);
    HIP_CHECK(hipDeviceSynchronize());

    // 拷贝结果回主机
    HIP_CHECK(hipMemcpy(h_grad_offset_gpu.data(),
                        d_grad_offset,
                        data_offset_size * sizeof(DATA_TYPE),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_mask_gpu.data(),
                        d_grad_mask,
                        data_mask_size * sizeof(DATA_TYPE),
                        hipMemcpyDeviceToHost));

#if 0
    std::cout << "验证CPP的GPU代码" << std::endl;
    std::vector<int> offset_shape = {
        batch_size, height_col, width_col, 2 * kernel_h * kernel_w * deformable_group};
    std::vector<int> mask_shape = {
        batch_size, height_col, width_col, kernel_h * kernel_w * deformable_group};
    if(VALIDATION_MODE)
    {
        save_tensor_to_file(h_grad_offset_gpu, offset_shape, "test_data_nhwc/grad_offset_cpp.bin");
        save_tensor_to_file(h_grad_mask_gpu, mask_shape, "test_data_nhwc/grad_mask_cpp.bin");
    }
#endif
#if 0
    bool offset_match = compare_results_fp16(h_grad_offset_cpu.data(),
                                             h_grad_offset_gpu.data(),
                                             std::min(1000, data_offset_size),
                                             "grad_offset",
                                             1e-2f);
    bool mask_match   = compare_results_fp16(h_grad_mask_cpu.data(),
                                           h_grad_mask_gpu.data(),
                                           std::min(1000, data_mask_size),
                                           "grad_mask",
                                           1e-2f);

    if(offset_match && mask_match)
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

    std::cout << "\n=== Sample Values (10 ~ 15) ===" << std::endl;
    for(int i = 0; i < std::min(5, data_offset_size); ++i)
    {
        // 将_Float16转换为float再输出
        std::cout << "grad_offset[" << i + 10
                  << "]: CPU=" << static_cast<float>(h_grad_offset_cpu[i + 10])
                  << ", GPU=" << static_cast<float>(h_grad_offset_gpu[i + 10]) << ", diff="
                  << std::abs(static_cast<float>(h_grad_offset_cpu[i + 10]) -
                              static_cast<float>(h_grad_offset_gpu[i + 10]))
                  << std::endl;
    }
#endif
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipStream_t stream;
    hipStreamCreate(&stream);
    const int iterations = 1;
    float total_ms       = 0.0f;

    for(int i = 0; i < iterations; ++i)
    {
        hipEventRecord(start, stream);
        hipLaunchKernelGGL(modulated_deformable_col2im_coord_gpu_kernel_nhwc,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           n,
                           d_grad_output,
                           d_weight,
                           d_data_im,
                           d_data_offset,
                           d_data_mask,
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
                           d_grad_offset,
                           d_grad_mask);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto avg_gpu_duration = total_ms / iterations;

    std::cout << "GPU time (avg over " << iterations << " runs): " << avg_gpu_duration << " ms"
              << std::endl;

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
