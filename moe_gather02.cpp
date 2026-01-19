#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void moe_gather_kernel(
    const hip_bfloat16 *scatter_tokens,
    const int *token_offset,
    const float *expert_weights,
    hip_bfloat16 *hidden_states,
    int num_shared_experts_per_rank,
    int num_tokens_offset,
    int num_tokens_per_rank,
    int allocated_tokens,
    int hidden_size,
    int topk)
{
    // 计算每个token需要处理的工作量
    int token_idx = blockIdx.x;
    int col = threadIdx.x;

    if (col >= hidden_size)
        return;

    // 处理共享专家部分
    for (int e = 0; e < num_shared_experts_per_rank; e++)
    {
        for (int t = 0; t < num_tokens_per_rank; t++)
        {
            int token_global = num_tokens_offset + t;
            if (token_global == token_idx)
            {
                int src_idx = (e * num_tokens_per_rank + t) * hidden_size + col;
                float token_val = static_cast<float>(scatter_tokens[src_idx]);

                // 直接累加，不乘权重
                float old_val = static_cast<float>(hidden_states[token_idx * hidden_size + col]);
                hidden_states[token_idx * hidden_size + col] = static_cast<hip_bfloat16>(old_val + token_val);
            }
        }
    }

    // 处理非共享专家部分
    for (int i = 0; i < allocated_tokens; i++)
    {
        int token_global = token_offset[i];
        if (token_global == token_idx)
        {
            // 计算权重索引k
            int tokens_per_k = allocated_tokens / topk;
            int k = i / tokens_per_k;

            int src_idx = (num_shared_experts_per_rank * num_tokens_per_rank + i) * hidden_size + col;
            float token_val = static_cast<float>(scatter_tokens[src_idx]);
            float weight = expert_weights[token_global * topk + k];
            float add_val = token_val * weight;

            // 累加到输出
            float old_val = static_cast<float>(hidden_states[token_idx * hidden_size + col]);
            hidden_states[token_idx * hidden_size + col] = static_cast<hip_bfloat16>(old_val + add_val);
        }
    }
}

void moe_gather(
    const hip_bfloat16 *scatter_tokens,
    const int *token_offset,
    const float *expert_weights,
    hip_bfloat16 *hidden_states,
    int num_shared_experts_per_rank,
    int num_tokens_offset,
    int num_tokens_per_rank,
    int allocated_tokens,
    int hidden_size,
    int topk,
    int num_tokens,
    hipStream_t stream)
{
    dim3 block_size(hidden_size);
    dim3 grid_size(num_tokens);

    hipLaunchKernelGGL(moe_gather_kernel,
                       grid_size, block_size, 0, stream,
                       scatter_tokens, token_offset, expert_weights, hidden_states,
                       num_shared_experts_per_rank, num_tokens_offset, num_tokens_per_rank,
                       allocated_tokens, hidden_size, topk);
}

double RAN_GEN(double A, double B)
{
    double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (B - A) + A;
    return r;
}

int main()
{
    const int world_size = 4;            // 总共4个rank
    const int num_tokens_per_rank = 100; // 当前rank负责的token数
    const int current_rank = 1;
    const int num_tokens_offset = num_tokens_per_rank * current_rank; // 当前rank的token偏移量，确定共享专家处理范围：[num_tokens_offset, num_tokens_offset + num_tokens_per_rank]
    const int allocated_tokens = 4;                                   // 分配的非共享token数
    const int hidden_size = 4;                                        // 隐藏层大小
    const int topk = 2;                                               // top-k值
    const int num_shared_experts_per_rank = 1;                        // 每个rank的共享专家数

    // 总token数 = 当前rank负责的token数 * 总的rank数
    const int num_tokens = world_size * num_tokens_per_rank;

    // 计算scatter_tokens大小: (共享专家数 * 当前rank的token数 + 分配的非共享token数) * hidden_size
    const int scatter_tokens_size = (num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens) * hidden_size;

    // 分配主机内存
    std::vector<hip_bfloat16> h_scatter_tokens(scatter_tokens_size);
    std::vector<int> h_token_offset(allocated_tokens);
    std::vector<float> h_expert_weights(num_tokens * topk);
    std::vector<hip_bfloat16> h_hidden_states(num_tokens * hidden_size, hip_bfloat16(0.0f));

    // 初始化数据
    // 1. 初始化scatter_tokens (共享专家部分 + 非共享专家部分)
    for (int i = 0; i < scatter_tokens_size; i++)
    {
        h_scatter_tokens[i] = hip_bfloat16(RAN_GEN(-1.0f, 1.0f));
    }

    // 2. 初始化token_offset (按topk分组)
    // 当前rank负责的全局token索引: [4,5,6,7]
    // h_token_offset[0] = num_tokens_offset + 0; // token 4 (k0)
    // h_token_offset[1] = num_tokens_offset + 1; // token 5 (k0)
    // h_token_offset[2] = num_tokens_offset + 2; // token 6 (k1)
    // h_token_offset[3] = num_tokens_offset + 3; // token 7 (k1)

    for (int t = 0; t < allocated_tokens; t++)
    {
        h_token_offset[t] = num_tokens_offset + t;
    }

    // 3. 初始化expert_weights
    for (int i = 0; i < num_tokens; i++)
    {
        for (int k = 0; k < topk; k++)
        {
            h_expert_weights[i * topk + k] = (k == 0) ? 0.6f : 0.4f;
        }
    }

    // 分配设备内存
    hip_bfloat16 *d_scatter_tokens;
    int *d_token_offset;
    float *d_expert_weights;
    hip_bfloat16 *d_hidden_states;

    hipMalloc(&d_scatter_tokens, scatter_tokens_size * sizeof(hip_bfloat16));
    hipMalloc(&d_token_offset, allocated_tokens * sizeof(int));
    hipMalloc(&d_expert_weights, num_tokens * topk * sizeof(float));
    hipMalloc(&d_hidden_states, num_tokens * hidden_size * sizeof(hip_bfloat16));

    // 拷贝数据到设备
    hipMemcpy(d_scatter_tokens, h_scatter_tokens.data(),
              scatter_tokens_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice);
    hipMemcpy(d_token_offset, h_token_offset.data(),
              allocated_tokens * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_expert_weights, h_expert_weights.data(),
              num_tokens * topk * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_hidden_states, 0, num_tokens * hidden_size * sizeof(hip_bfloat16));

    // 创建HIP stream
    hipStream_t stream;
    hipStreamCreate(&stream);

    // 调用moe_gather函数
    moe_gather(
        d_scatter_tokens,
        d_token_offset,
        d_expert_weights,
        d_hidden_states,
        num_shared_experts_per_rank,
        num_tokens_offset,
        num_tokens_per_rank,
        allocated_tokens,
        hidden_size,
        topk,
        num_tokens,
        stream);

    // 同步等待kernel完成
    hipStreamSynchronize(stream);

    // 拷贝结果回主机
    hipMemcpy(h_hidden_states.data(), d_hidden_states,
              num_tokens * hidden_size * sizeof(hip_bfloat16), hipMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Results:\n";
    for (int i = 0; i < num_tokens; i++)
    {
        std::cout << "Token " << i << ": [";
        for (int j = 0; j < hidden_size; j++)
        {
            std::cout << static_cast<float>(h_hidden_states[i * hidden_size + j]);
            if (j < hidden_size - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // 验证结果
    bool success = true;
    const float tolerance = 1e-2f;
    std::vector<float> expected(num_tokens * hidden_size, 0.0f);

    for (int e = 0; e < num_shared_experts_per_rank; e++)
    {
        for (int t = 0; t < num_tokens_per_rank; t++)
        {
            int token_global = num_tokens_offset + t;
            for (int h = 0; h < hidden_size; h++)
            {
                int src_idx = (e * num_tokens_per_rank + t) * hidden_size + h;
                expected[token_global * hidden_size + h] += static_cast<float>(h_scatter_tokens[src_idx]);
            }
        }
    }

    for (int i = 0; i < allocated_tokens; i++)
    {
        int token_global = h_token_offset[i];
        for (int k = 0; k < topk; k++)
        {
            float weight = h_expert_weights[token_global * topk + k];

            for (int h = 0; h < hidden_size; h++)
            {
                int src_idx = (num_shared_experts_per_rank * num_tokens_per_rank + i) * hidden_size + h;

                expected[token_global * hidden_size + h] += static_cast<float>(h_scatter_tokens[src_idx]) * weight;
            }
        }
    }

    /*
    */

    // 比较结果
    for (int i = 0; i < num_tokens * hidden_size; i++)
    {
        float actual = static_cast<float>(h_hidden_states[i]);
        float exp_val = expected[i];

        if (fabs(actual - exp_val) > tolerance)
        {
            std::cout << "Mismatch at index " << i << ": expected " << exp_val
                      << ", got " << actual << std::endl;
            success = false;
        }
    }

    std::cout << "\nVerification: " << (success ? "PASSED" : "FAILED") << std::endl;

    // 释放资源
    hipFree(d_scatter_tokens);
    hipFree(d_token_offset);
    hipFree(d_expert_weights);
    hipFree(d_hidden_states);
    hipStreamDestroy(stream);

    return 0;
}
