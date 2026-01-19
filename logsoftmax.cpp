extern "C" __global__ void __launch_bounds__(THREADS)
    SoftMaxCommon(const float* in,
                  OUT_TYPE* out,
                  float* M,
                  float* Z,
                  float* bias,
                  float* Amax,
                  const float descale_Q,
                  const float descale_K,
                  const float scale_S,
                  const uint64_t seed,
                  const uint64_t offset,
                  const float dropout_P,
                  uint32_t seq_len,
                  uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid          = threadIdx.x;
    const uint32_t laneId       = lid % warpSize;
    const uint32_t warpId       = lid / warpSize;
    const float descaler        = (descale_Q ? descale_Q : 1.0f) * (descale_K ? descale_K : 1.0f);
    const float dropout         = (dropout_P && seed && offset) ? (dropout_P) : 0.0f;
    const float scaler          = (scale_S ? scale_S : 1.0f) / (1.0f - dropout);
    const bool save_stats       = M && Z && (lid == 0);

    rocrand_state_xorwow rng;
    if(dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // rocrand_init(prng::hash(seed + idx), 0, offset, &rng);
        rocrand_init(seed + idx, 0, offset, &rng);
    }

    float r_Amax = 0;

    // uint32_t stride = blockDim.x * gridDim.x;
    // for(uint64_t gid = blockIdx.x; gid < nhs; gid += stride)
    for(uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float* line      = in + gid * seq_len;
        const float* bias_line = bias + gid * seq_len;
        auto res               = out + gid * seq_len;

        float r_max = reductionCommon<NumWarps>(
            line,
            bias_line,
            std::numeric_limits<float>::lowest(),
            seq_len,
            fmaxf_op,
            [descaler](float x) { return x * descaler; },
            lid,
            laneId,
            warpId);

        float r_sum = 1.0f / reductionCommon<NumWarps>(    //   1 / 归一化因子
                                 line,
                                 nullptr,
                                 0,
                                 seq_len,
                                 plus_op,
                                 [r_max, descaler](float x) { return expf(x * descaler - r_max); },
                                 lid,
                                 laneId,
                                 warpId);
        
        // const uint32_t stride = blockDim.x * gridDim.x;
        // for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += stride)
        for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += blockDim.x)
        {
            float local_val = expf(line[loop_lid] * descaler - r_max) * r_sum;

            // It is supposed to be maximum of absolute values,
            // however we do not need abs() because expf() above produces
            // non-negative value. Plain max() is enough.
            r_Amax = fmaxf_op(r_Amax, local_val);

            res[loop_lid] =
                static_cast<OUT_TYPE>(doDropout(dropout, &rng) ? 0.0f : local_val * scaler);
        }

        if(save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if(Amax)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
        if(lid == 0)
        {
            atomicMaxOfNonNegative(Amax, r_Amax);
        }
    }
    return;
}


#define UNROLL_FACTOR 8

extern "C" __global__ void __launch_bounds__(THREADS)
    SoftMaxCommon(const float *in,
                  OUT_TYPE *out,
                  float *M,
                  float *Z,
                  float *bias,
                  float *Amax,
                  const float descale_Q,
                  const float descale_K,
                  const float scale_S,
                  const uint64_t seed,
                  const uint64_t offset,
                  const float dropout_P,
                  uint32_t seq_len,
                  uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid = threadIdx.x;
    const uint32_t laneId = lid % warpSize;
    const uint32_t warpId = lid / warpSize;
    const float descaler = (descale_Q ? descale_Q : 1.0f) * (descale_K ? descale_K : 1.0f);
    const float dropout = (dropout_P && seed && offset) ? (dropout_P) : 0.0f;
    const float scaler = (scale_S ? scale_S : 1.0f) / (1.0f - dropout);
    const bool save_stats = M && Z && (lid == 0);

    rocrand_state_xorwow rng;
    if (dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // rocrand_init(prng::hash(seed + idx), 0, offset, &rng);
        rocrand_init(seed + idx, 0, offset, &rng);
    }

    float r_Amax = 0;

    uint32_t seq_len_rounded = ((seq_len + UNROLL_FACTOR - 1) / UNROLL_FACTOR) * UNROLL_FACTOR;

    // uint32_t stride = blockDim.x * gridDim.x;
    // for(uint64_t gid = blockIdx.x; gid < nhs; gid += stride)
    for (uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float *line = in + gid * seq_len;
        const float *bias_line = bias + gid * seq_len;
        auto res = out + gid * seq_len;

        float r_max = reductionCommon<NumWarps>(
            line,
            bias_line,
            std::numeric_limits<float>::lowest(),
            seq_len,
            fmaxf_op,
            [descaler](float x)
            { return x * descaler; },
            lid,
            laneId,
            warpId);

        float r_sum = 1.0f / reductionCommon<NumWarps>( //   1 / 归一化因子
                                 line,
                                 nullptr,
                                 0,
                                 seq_len,
                                 plus_op,
                                 [r_max, descaler](float x)
                                 { return expf(x * descaler - r_max); },
                                 lid,
                                 laneId,
                                 warpId);

        

        // const uint32_t stride = blockDim.x * gridDim.x;
        // for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += stride)
        for (uint32_t base_idx = lid * UNROLL_FACTOR; // 每个线程的起始索引 = 线程ID * UNROLL_FACTOR
             base_idx < seq_len_rounded;                      // 确保不超过序列长度
             base_idx += blockDim.x * UNROLL_FACTOR)  // 步长 = 线程块总线程数 * 展开因子
        {
            const float* line_ptr = line + base_idx;
            float reg_array[UNROLL_FACTOR];
            
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++)
            {
                reg_array[u] = line_ptr[u];
            }

            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++)
            {
                const uint32_t idx = base_idx + u; // 连续处理：base_idx+0, base_idx+1, ..., base_idx+UNROLL_FACTOR-1
                if (idx < seq_len)
                {
                    float x = reg_array[u] * descaler - r_max;
                    float local_val = expf(x) * r_sum;
                    r_Amax = fmaxf(r_Amax, local_val);
                    res[idx] = static_cast<OUT_TYPE>(doDropout(dropout, &rng) ? 0.0f : local_val * scaler);
                }
            }
        }

        // 处理余数部分（若seq_len_rounded != seq_len）
        // if (seq_len_rounded > seq_len && lid == 0) 
        // {
        //     for (uint32_t idx = seq_len; idx < seq_len_rounded; idx++) 
        //     {
        //         res[idx] = 0.0f; // 填充无效数据
        //     }
        // }

        if (save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if (Amax)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
        if (lid == 0)
        {
            atomicMaxOfNonNegative(Amax, r_Amax);
        }
    }
    return;
}


#define UNROLL_FACTOR 8

extern "C" __global__ void __launch_bounds__(THREADS)
    SoftMaxCommon(const float *in,
                  OUT_TYPE *out,
                  float *M,
                  float *Z,
                  float *bias,
                  float *Amax,
                  const float descale_Q,
                  const float descale_K,
                  const float scale_S,
                  const uint64_t seed,
                  const uint64_t offset,
                  const float dropout_P,
                  uint32_t seq_len,
                  uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid = threadIdx.x;
    const uint32_t laneId = lid % warpSize;
    const uint32_t warpId = lid / warpSize;
    const float descaler = (descale_Q ? descale_Q : 1.0f) * (descale_K ? descale_K : 1.0f);
    const float dropout = (dropout_P && seed && offset) ? (dropout_P) : 0.0f;
    const float scaler = (scale_S ? scale_S : 1.0f) / (1.0f - dropout);
    const bool save_stats = M && Z && (lid == 0);

    rocrand_state_xorwow rng;
    if (dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // rocrand_init(prng::hash(seed + idx), 0, offset, &rng);
        rocrand_init(seed + idx, 0, offset, &rng);
    }

    float r_Amax = 0;

    uint32_t seq_len_rounded = ((seq_len + UNROLL_FACTOR - 1) / UNROLL_FACTOR) * UNROLL_FACTOR;

    // uint32_t stride = blockDim.x * gridDim.x;
    // for(uint64_t gid = blockIdx.x; gid < nhs; gid += stride)
    for (uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float *line = in + gid * seq_len;
        const float *bias_line = bias + gid * seq_len;
        auto res = out + gid * seq_len;

        float r_max = reductionCommon<NumWarps>(
            line,
            bias_line,
            std::numeric_limits<float>::lowest(),
            seq_len,
            fmaxf_op,
            [descaler](float x)
            { return x * descaler; },
            lid,
            laneId,
            warpId);

        float r_sum = 1.0f / reductionCommon<NumWarps>( //   1 / 归一化因子
                                 line,
                                 nullptr,
                                 0,
                                 seq_len,
                                 plus_op,
                                 [r_max, descaler](float x)
                                 { return expf(x * descaler - r_max); },
                                 lid,
                                 laneId,
                                 warpId);

        

        // const uint32_t stride = blockDim.x * gridDim.x;
        // for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += stride)
        for (uint32_t base_idx = lid * UNROLL_FACTOR; // 每个线程的起始索引 = 线程ID * UNROLL_FACTOR
             base_idx < seq_len_rounded;                      // 确保不超过序列长度
             base_idx += blockDim.x * UNROLL_FACTOR)  // 步长 = 线程块总线程数 * 展开因子
        {
            const float* line_ptr = line + base_idx;

            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++)
            {
                const uint32_t idx = base_idx + u; // 连续处理：base_idx+0, base_idx+1, ..., base_idx+UNROLL_FACTOR-1
                if (idx < seq_len)
                {
                    float x = line_ptr[u] * descaler - r_max;
                    float local_val = expf(x) * r_sum;
                    r_Amax = fmaxf(r_Amax, local_val);
                    res[idx] = static_cast<OUT_TYPE>(doDropout(dropout, &rng) ? 0.0f : local_val * scaler);
                }
            }
        }

        if (save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if (Amax)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
        if (lid == 0)
        {
            atomicMaxOfNonNegative(Amax, r_Amax);
        }
    }
    return;
}

这两个kernel哪个性能好，好在哪里，为什么好，性能提升点在哪里

/*
粗化线程
指令级的并行（访存和计算）



寄存器缓存，减少访问全局内存的次数，这里说的有疑问，这里也是从全局内存一个一个加载至寄存器的，不存在减少访问全局的次数吧？
第一个kernel内存访问集中，第二个kernel内存访问分散，第一个更容易合并访存。


寄存器中重复使用reg_array数据，避免重复计算（如descaler乘法）。，为什么说寄存器中重复使用reg_array数据



访存合并（Memory Coalescing）是GPU优化内存访问的关键技术。其核心思想是：
当线程束（warp）内的线程访问连续的全局内存地址时，硬件可以将多个内存请求合并为更少的事务，从而减少内存带宽浪费。合并效率取决于以下条件：

连续性：线程束内线程访问的地址必须是连续的。

对齐性：内存地址需对齐到硬件支持的事务粒度（如128字节）。

访问模式：线程ID与内存地址的映射关系需满足合并规则（如线程0访问地址0，线程1访问地址1，以此类推）。
*/


#define UNROLL_FACTOR 8

extern "C" __global__ void __launch_bounds__(THREADS)
    SoftMaxCommon(const float *in,
                  OUT_TYPE *out,
                  float *M,
                  float *Z,
                  float *bias,
                  float *Amax,
                  const float descale_Q,
                  const float descale_K,
                  const float scale_S,
                  const uint64_t seed,
                  const uint64_t offset,
                  const float dropout_P,
                  uint32_t seq_len,
                  uint64_t nhs)
{
    static_assert(THREADS % warpSize == 0);
    constexpr uint32_t NumWarps = THREADS / warpSize;
    const uint32_t lid = threadIdx.x;
    const uint32_t laneId = lid % warpSize;
    const uint32_t warpId = lid / warpSize;
    const float descaler = (descale_Q ? descale_Q : 1.0f) * (descale_K ? descale_K : 1.0f);
    const float dropout = (dropout_P && seed && offset) ? (dropout_P) : 0.0f;
    const float scaler = (scale_S ? scale_S : 1.0f) / (1.0f - dropout);
    const bool save_stats = M && Z && (lid == 0);

    rocrand_state_xorwow rng;
    if (dropout > 0.0f)
    {
        const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // rocrand_init(prng::hash(seed + idx), 0, offset, &rng);
        rocrand_init(seed + idx, 0, offset, &rng);
    }

    float r_Amax = 0;

    uint32_t seq_len_rounded = ((seq_len + UNROLL_FACTOR - 1) / UNROLL_FACTOR) * UNROLL_FACTOR;

    // uint32_t stride = blockDim.x * gridDim.x;
    // for(uint64_t gid = blockIdx.x; gid < nhs; gid += stride)
    for (uint64_t gid = blockIdx.x; gid < nhs; gid += gridDim.x)
    {
        const float *line = in + gid * seq_len;
        const float *bias_line = bias + gid * seq_len;
        auto res = out + gid * seq_len;

        float r_max = reductionCommon<NumWarps>(
            line,
            bias_line,
            std::numeric_limits<float>::lowest(),
            seq_len,
            fmaxf_op,
            [descaler](float x)
            { return x * descaler; },
            lid,
            laneId,
            warpId);

        float r_sum = 1.0f / reductionCommon<NumWarps>( //   1 / 归一化因子
                                 line,
                                 nullptr,
                                 0,
                                 seq_len,
                                 plus_op,
                                 [r_max, descaler](float x)
                                 { return expf(x * descaler - r_max); },
                                 lid,
                                 laneId,
                                 warpId);

        

        // const uint32_t stride = blockDim.x * gridDim.x;
        // for(uint32_t loop_lid = lid; loop_lid < seq_len; loop_lid += stride)
        // 数据加载与计算（向量化+线程粗化）
        for (uint32_t base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL_FACTOR; 
             base_idx < seq_len_rounded; 
             base_idx += gridDim.x * blockDim.x * UNROLL_FACTOR) 
        {
            const float* line_ptr = line + base_idx;
            float reg_array[UNROLL_FACTOR];

            // 向量化加载
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u += VEC_SIZE) {
                float4 vec_data = *reinterpret_cast<const float4*>(line_ptr + u);
                reg_array[u + 0] = vec_data.x;
                reg_array[u + 1] = vec_data.y;
                reg_array[u + 2] = vec_data.z;
                reg_array[u + 3] = vec_data.w;
                // 若UNROLL_FACTOR > 4，继续加载后续vec_data
            }

            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++)
            {
                const uint32_t idx = base_idx + u; // 连续处理：base_idx+0, base_idx+1, ..., base_idx+UNROLL_FACTOR-1
                if (idx < seq_len)
                {
                    float x = reg_array[u] * descaler - r_max;
                    float local_val = expf(x) * r_sum;
                    r_Amax = fmaxf(r_Amax, local_val);
                    res[idx] = static_cast<OUT_TYPE>(doDropout(dropout, &rng) ? 0.0f : local_val * scaler);
                }
            }
        }

        // 处理余数部分（若seq_len_rounded != seq_len）
        // if (seq_len_rounded > seq_len && lid == 0) 
        // {
        //     for (uint32_t idx = seq_len; idx < seq_len_rounded; idx++) 
        //     {
        //         res[idx] = 0.0f; // 填充无效数据
        //     }
        // }

        if (save_stats)
        {
            M[gid] = r_max;
            Z[gid] = r_sum;
        }
    }

    if (Amax)
    {
        r_Amax = reductionBlock<NumWarps>(r_Amax, fmaxf_op, lid, laneId, warpId);
        if (lid == 0)
        {
            atomicMaxOfNonNegative(Amax, r_Amax);
        }
    }
    return;
}