#include <hip/hip_runtime.h>
#include <cmath>
#include <iostream>

// 常量定义
constexpr int BLOCK_DIM         = 64; // KV块大小
constexpr int HEAD_DIM          = 64; // 头维度大小
constexpr int THREADS_PER_BLOCK = 256;

// 核心KV分块实现 - 使用grid.z维度处理KV块
__global__ void
flash_attention_v2_kernel(const half* Q, // [batch_size, num_heads, seq_len_q, head_dim]
                          const half* K, // [batch_size, num_heads, seq_len_kv, head_dim]
                          const half* V, // [batch_size, num_heads, seq_len_kv, head_dim]
                          half* O,  // 输出矩阵 [batch_size, num_heads, seq_len_q, head_dim]
                          float* L, // 最大值统计量 [batch_size, num_heads, seq_len_q]
                          float* M, // 指数和统计量 [batch_size, num_heads, seq_len_q]
                          const int batch_size,
                          const int num_heads,
                          const int seq_len_q,
                          const int seq_len_kv)
{
    // 每个线程块处理:
    //   grid.x: batch 维度
    //   grid.y: head 维度
    //   grid.z: KV块索引

    const int batch_idx    = blockIdx.x;
    const int head_idx     = blockIdx.y;
    const int kv_block_idx = blockIdx.z;

    // 计算当前KV块的范围
    const int kv_start  = kv_block_idx * BLOCK_DIM;
    const int kv_end    = min(kv_start + BLOCK_DIM, seq_len_kv);
    const int kv_length = kv_end - kv_start;

    // 计算Q块索引 (每个线程块处理一个Q块)
    const int q_block_idx = threadIdx.y;
    const int q_start     = q_block_idx * BLOCK_DIM;
    const int q_end       = min(q_start + BLOCK_DIM, seq_len_q);
    const int q_length    = q_end - q_start;

    // 共享内存存储 - 用于KV块
    __shared__ half shared_K[BLOCK_DIM][HEAD_DIM];
    __shared__ half shared_V[BLOCK_DIM][HEAD_DIM];

    // 线程局部的Q块存储
    half local_Q[BLOCK_DIM][HEAD_DIM];

    // 加载Q块到线程局部内存
    if(threadIdx.x < HEAD_DIM)
    {
        for(int i = 0; i < q_length; i++)
        {
            const long index = ((batch_idx * num_heads + head_idx) * seq_len_q * HEAD_DIM) +
                               ((q_start + i) * HEAD_DIM) + threadIdx.x;
            local_Q[i][threadIdx.x] = Q[index];
        }
    }

    // 加载KV块到共享内存
    if(threadIdx.x < kv_length)
    {
        for(int d = 0; d < HEAD_DIM; d++)
        {
            const long k_index = ((batch_idx * num_heads + head_idx) * seq_len_kv * HEAD_DIM) +
                                 ((kv_start + threadIdx.x) * HEAD_DIM) + d;
            const long v_index = ((batch_idx * num_heads + head_idx) * seq_len_kv * HEAD_DIM) +
                                 ((kv_start + threadIdx.x) * HEAD_DIM) + d;

            shared_K[threadIdx.x][d] = K[k_index];
            shared_V[threadIdx.x][d] = V[v_index];
        }
    }
    __syncthreads();

    // 线程局部的输出和统计量
    float thread_O[BLOCK_DIM][HEAD_DIM] = {0};
    float thread_M[BLOCK_DIM]           = {0};         // 指数和
    float thread_L[BLOCK_DIM]           = {-INFINITY}; // 最大值

    // 处理当前(Q块, KV块)对
    for(int i = 0; i < q_length; i++)
    {
        // 计算QK^T
        float S[BLOCK_DIM] = {0};
        for(int j = 0; j < kv_length; j++)
        {
            float dot = 0.0f;
            for(int d = 0; d < HEAD_DIM; d++)
            {
                dot += __half2float(local_Q[i][d]) * __half2float(shared_K[j][d]);
            }
            S[j] = dot;
        }

        // 在线softmax更新
        // 1. 计算当前KV块的最大值
        float m_current = -INFINITY;
        for(int j = 0; j < kv_length; j++)
        {
            if(S[j] > m_current)
                m_current = S[j];
        }

        // 2. 更新全局最大值
        float m_prev = thread_L[i];
        float m_new  = fmaxf(m_prev, m_current);
        thread_L[i]  = m_new;

        // 3. 计算指数和
        float exp_sum      = 0.0f;
        float P[BLOCK_DIM] = {0};
        for(int j = 0; j < kv_length; j++)
        {
            P[j] = expf(S[j] - m_new);
            exp_sum += P[j];
        }

        // 4. 更新指数和统计
        float l_prev = thread_M[i];
        float l_new  = l_prev * expf(m_prev - m_new) + exp_sum;
        thread_M[i]  = l_new;

        // 5. 更新输出
        // 5.1 调整历史输出
        float scale = expf(m_prev - m_new);
        for(int d = 0; d < HEAD_DIM; d++)
        {
            thread_O[i][d] *= scale;
        }

        // 5.2 添加当前KV块的贡献
        for(int j = 0; j < kv_length; j++)
        {
            float p = P[j];
            for(int d = 0; d < HEAD_DIM; d++)
            {
                thread_O[i][d] += p * __half2float(shared_V[j][d]);
            }
        }
    }

    // 原子更新全局统计量和输出
    for(int i = 0; i < q_length; i++)
    {
        const int q_index     = q_start + i;
        const long global_idx = (batch_idx * num_heads + head_idx) * seq_len_q + q_index;

        // 原子更新最大值
        float old_L =
            atomicMax(reinterpret_cast<int*>(&L[global_idx]), __float_as_int(thread_L[i]));
        float prev_L = __int_as_float(old_L);

        // 原子更新指数和
        float new_M = thread_M[i] * expf(thread_L[i] - prev_L);
        atomicAdd(&M[global_idx], new_M);

        // 原子更新输出
        for(int d = 0; d < HEAD_DIM; d++)
        {
            const long o_index = global_idx * HEAD_DIM + d;
            float scale        = expf(thread_L[i] - prev_L);
            atomicAdd(&O[o_index], thread_O[i][d] * scale);
        }
    }
}

// 最终归一化内核
__global__ void
normalize_kernel(half* O, const float* M, int batch_size, int num_heads, int seq_len_q)
{
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int q_idx     = threadIdx.x + blockIdx.z * blockDim.x;

    if(q_idx >= seq_len_q)
        return;

    const long global_idx = (batch_idx * num_heads + head_idx) * seq_len_q + q_idx;
    const float norm      = 1.0f / M[global_idx];

    for(int d = 0; d < HEAD_DIM; d++)
    {
        const long o_index = global_idx * HEAD_DIM + d;
        O[o_index]         = __float2half(__half2float(O[o_index]) * norm);
    }
}

// 封装函数
void flash_attention_v2(const half* Q,
                        const half* K,
                        const half* V,
                        half* O,
                        int batch_size,
                        int num_heads,
                        int seq_len_q,
                        int seq_len_kv)
{
    // 计算KV块数量
    const int num_kv_blocks = (seq_len_kv + BLOCK_DIM - 1) / BLOCK_DIM;
    const int num_q_blocks  = (seq_len_q + BLOCK_DIM - 1) / BLOCK_DIM;

    // 分配统计量内存
    float *L, *M;
    cudaMalloc(&L, batch_size * num_heads * seq_len_q * sizeof(float));
    cudaMalloc(&M, batch_size * num_heads * seq_len_q * sizeof(float));
    cudaMemset(L, 0, batch_size * num_heads * seq_len_q * sizeof(float));
    cudaMemset(M, 0, batch_size * num_heads * seq_len_q * sizeof(float));

    // 初始化输出
    cudaMemset(O, 0, batch_size * num_heads * seq_len_q * HEAD_DIM * sizeof(half));

    // 配置线程块
    dim3 grid(batch_size, num_heads, num_kv_blocks);
    dim3 block(HEAD_DIM, num_q_blocks);

    // 调用注意力内核
    flash_attention_v2_kernel<<<grid, block>>>(
        Q, K, V, O, L, M, batch_size, num_heads, seq_len_q, seq_len_kv);

    // 配置归一化内核
    dim3 norm_grid(batch_size, num_heads, (seq_len_q + 255) / 256);
    dim3 norm_block(256);

    // 调用归一化内核
    normalize_kernel<<<norm_grid, norm_block>>>(O, M, batch_size, num_heads, seq_len_q);

    // 清理
    cudaDeviceSynchronize();
    cudaFree(L);
    cudaFree(M);
}
