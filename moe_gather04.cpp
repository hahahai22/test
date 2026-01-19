#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>

__global__ void moe_gather_kernel(const hip_bfloat16* scatter_tokens, // in
                                  const int real_scatter_tokens,
                                  const int hidden_size,
                                  const int32* scatter_tokens_offset,
                                  const hip_bfloat16* convergent_tokens, // out
                                  const int num_tokens,
                                  hipStream_t stream)
{
    int col = threadIdx.x;
    int token_idx = blockIdx.x;

    int src_token_idx = scatter_tokens_offset[token_idx];

    for (int h = col; h < hidden_size; h += blockDim.x)
    {
        int src_idx = scatter_tokens_offset[src_token_idx] * hidden_size + h;


        int dst_idx = token_idx * hidden_size + h;
        convergent_tokens[dst_idx] += scatter_tokens[src_idx];
    }
}

void moe_gather(const hip_bfloat16* scatter_tokens, // in
                const int real_scatter_tokens,
                const int hidden_size,
                const int32* scatter_tokens_offset,
                const hip_bfloat16* convergent_tokens, // out
                const int num_tokens,
                hipStream_t stream)
{
    dim3 block_size(256);
    dim3 grid_size(real_scatter_tokens);

    hipLaunchKernelGGL(moe_gather_kernel,
                       grid_size,
                       block_size,
                       0,
                       stream,
                       scatter_tokens,
                       real_scatter_tokens,
                       hidden_size,
                       scatter_tokens_offset,
                       convergent_tokens,
                       num_tokens);
}

double RAN_GEN(double A, double B)
{
    double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (B - A) + A;
    return r;
}

float random_float(float min = -1.0f, float max = 1.0f)
{
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

int main()
{
    /*
    word_size = DP * SP * EP;  // DP * SP  = EP;

    */
    const int world_size          = 2;
    const int num_tokens_per_rank = 6;
    const int topk                = 2;
    const int current_rank        = 1;
    const int num_tokens_offset =
        num_tokens_per_rank *
        current_rank; 
                      
    const int allocated_tokens = num_tokens_per_rank * topk; // 分配的非共享token数
    const int hidden_size      = 4;                          // 隐藏层大小
    const int num_shared_experts_per_rank = 2;               // 每个rank的共享专家数

    // 总token数 = 当前rank负责的token数 * 总的rank数
    const int num_tokens = world_size * num_tokens_per_rank;

    // 计算scatter_tokens大小: (共享专家数 * 当前rank的token数 + 分配的非共享token数) * hidden_size
    const int scatter_tokens_size =
        (num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens) * hidden_size;

    // 分配主机内存
    std::vector<hip_bfloat16> h_scatter_tokens(scatter_tokens_size);
    std::vector<int> h_token_offset(allocated_tokens);
    std::vector<float> h_expert_weights(num_tokens * topk);
    std::vector<hip_bfloat16> h_hidden_states(num_tokens * hidden_size, hip_bfloat16(0.0f));

    // 初始化数据
    // 1. 初始化scatter_tokens (共享专家部分 + 非共享专家部分)
    for(int i = 0; i < scatter_tokens_size; i++)
    {
        h_scatter_tokens[i] = hip_bfloat16(RAN_GEN(-1.0f, 1.0f));
    }

    // 2. 初始化token_offset (按topk分组)
    // 当前rank负责的全局token索引: [4,5,6,7]
    h_token_offset[0]  = 8;  // token 0 (t0k0)
    h_token_offset[1]  = 3;  // token 0 (t0k1)
    h_token_offset[2]  = 4;  // token 1 (t1k0)
    h_token_offset[3]  = 9;  // token 1 (t1k1)
    h_token_offset[4]  = 0;  // token 2 (t2k0)
    h_token_offset[5]  = 10; // token 2 (t2k1)
    h_token_offset[6]  = 1;  // token 3 (t3k0)
    h_token_offset[7]  = 5;  // token 3 (t3k1)
    h_token_offset[8]  = 11; // token 4 (t4k0)
    h_token_offset[9]  = 6;  // token 4 (t4k1)
    h_token_offset[10] = 7;  // token 5 (t5k0)
    h_token_offset[11] = 2;  // token 5 (t5k1)

    // int tokens_per_k = allocated_tokens / topk;
    // for (int i = 0; i < allocated_tokens; i++)
    // {
    //     int k = i / tokens_per_k;
    //     int token_index_in_rank = i % num_tokens_per_rank;
    //     h_token_offset[i] = num_tokens_offset + token_index_in_rank;
    // }

    // 3. 初始化expert_weights
    // for (int i = 0; i < num_tokens; i++)
    // {
    //     for (int k = 0; k < topk; k++)
    //     {
    //         h_expert_weights[i * topk + k] = (k == 0) ? 0.6f : 0.4f;
    //     }
    // }

    for(int i = 0; i < num_tokens; i++)
    {
        // 归一化权重
        float sum = 0.0f;
        for(int k = 0; k < topk; k++)
        {
            h_expert_weights[i * topk + k] = random_float(0.1f, 0.9f);
            sum += h_expert_weights[i * topk + k];
        }

        // 归一化
        for(int k = 0; k < topk; k++)
        {
            h_expert_weights[i * topk + k] /= sum;
        }
    }

    // 分配设备内存
    hip_bfloat16* d_scatter_tokens;
    int* d_token_offset;
    float* d_expert_weights;
    hip_bfloat16* d_hidden_states;

    hipMalloc(&d_scatter_tokens, scatter_tokens_size * sizeof(hip_bfloat16));
    hipMalloc(&d_token_offset, allocated_tokens * sizeof(int));
    hipMalloc(&d_expert_weights, num_tokens * topk * sizeof(float));
    hipMalloc(&d_hidden_states, num_tokens * hidden_size * sizeof(hip_bfloat16));

    // 拷贝数据到设备
    hipMemcpy(d_scatter_tokens,
              h_scatter_tokens.data(),
              scatter_tokens_size * sizeof(hip_bfloat16),
              hipMemcpyHostToDevice);
    hipMemcpy(d_token_offset,
              h_token_offset.data(),
              allocated_tokens * sizeof(int),
              hipMemcpyHostToDevice);
    hipMemcpy(d_expert_weights,
              h_expert_weights.data(),
              num_tokens * topk * sizeof(float),
              hipMemcpyHostToDevice);
    hipMemset(d_hidden_states, 0, num_tokens * hidden_size * sizeof(hip_bfloat16));

    // 创建HIP stream
    hipStream_t stream;
    hipStreamCreate(&stream);

    // 调用moe_gather函数
    moe_gather(d_scatter_tokens,
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
    hipMemcpy(h_hidden_states.data(),
              d_hidden_states,
              num_tokens * hidden_size * sizeof(hip_bfloat16),
              hipMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Results:\n";
    for(int i = 0; i < num_tokens; i++)
    {
        std::cout << "Token " << i << ": [";
        for(int j = 0; j < hidden_size; j++)
        {
            std::cout << static_cast<float>(h_hidden_states[i * hidden_size + j]);
            if(j < hidden_size - 1)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // 验证结果
    bool success          = true;
    const float tolerance = 1e-2f;
    std::vector<float> expected(num_tokens * hidden_size, 0.0f);

    // 计算期望结果
    

    // 比较结果
    for(int i = num_tokens_offset; i < num_tokens * hidden_size; i++)
    {
        float actual  = static_cast<float>(h_hidden_states[i]);
        float exp_val = expected[i];

        if(fabs(actual - exp_val) > tolerance)
        {
            std::cout << "Mismatch at index " << i << ": expected " << exp_val << ", got " << actual
                      << std::endl;
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
