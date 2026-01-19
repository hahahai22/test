#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        hipError_t error = (cmd);                                                              \
        if(error != hipSuccess)                                                                \
        {                                                                                      \
            std::cerr << "HIP error (" << hipGetErrorString(error) << ") at line " << __LINE__ \
                      << " in file " << __FILE__ << "\n";                                      \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

__global__ void moe_gather_kernel(const hip_bfloat16* scatter_tokens,
                                  const int* token_offset,
                                  const float* expert_weights,
                                  hip_bfloat16* hidden_states,
                                  int num_shared_experts_per_rank,
                                  int num_tokens_offset,
                                  int num_tokens_per_rank,
                                  int allocated_tokens,
                                  int hidden_size,
                                  int topk,
                                  int num_tokens)
{
    int token_idx = blockIdx.x;
    int col       = threadIdx.x;

    if(token_idx >= num_tokens || col >= hidden_size)
        return;

    float result = 0.0f;

    // 处理共享专家部分 - 直接累加，不乘权重
    if(token_idx >= num_tokens_offset && token_idx < num_tokens_offset + num_tokens_per_rank)
    {
        int local_token_idx = token_idx - num_tokens_offset;

        for(int e = 0; e < num_shared_experts_per_rank; e++)
        {
            int src_idx     = (e * num_tokens_per_rank + local_token_idx) * hidden_size + col;
            float token_val = static_cast<float>(scatter_tokens[src_idx]);
            result += token_val;
        }
    }

    // 处理非共享专家部分 - 乘以权重后累加
    for(int i = 0; i < allocated_tokens; i++)
    {
        if(token_offset[i] == token_idx)
        {
            // 计算权重索引k
            int tokens_per_k = allocated_tokens / topk;
            int k            = i / tokens_per_k;

            int src_idx =
                (num_shared_experts_per_rank * num_tokens_per_rank + i) * hidden_size + col;
            float token_val = static_cast<float>(scatter_tokens[src_idx]);
            float weight    = expert_weights[token_idx * topk + k];

            result += token_val * weight;
        }
    }

    hidden_states[token_idx * hidden_size + col] = static_cast<hip_bfloat16>(result);
}

void moe_gather(const hip_bfloat16* scatter_tokens,
                const int* token_offset,
                const float* expert_weights,
                hip_bfloat16* hidden_states,
                int num_shared_experts_per_rank,
                int num_tokens_offset,
                int num_tokens_per_rank,
                int allocated_tokens,
                int hidden_size,
                int topk,
                int num_tokens,
                hipStream_t stream)
{
    dim3 block(hidden_size);
    dim3 grid(num_tokens);

    hipLaunchKernelGGL(moe_gather_kernel,
                       grid,
                       block,
                       0,
                       stream,
                       scatter_tokens,
                       token_offset,
                       expert_weights,
                       hidden_states,
                       num_shared_experts_per_rank,
                       num_tokens_offset,
                       num_tokens_per_rank,
                       allocated_tokens,
                       hidden_size,
                       topk,
                       num_tokens);
}

void cpu_moe_gather(const std::vector<hip_bfloat16>& scatter_tokens,
                    const std::vector<int>& token_offset,
                    const std::vector<float>& expert_weights,
                    std::vector<hip_bfloat16>& hidden_states,
                    int num_shared_experts_per_rank,
                    int num_tokens_offset,
                    int num_tokens_per_rank,
                    int allocated_tokens,
                    int hidden_size,
                    int topk,
                    int num_tokens)
{
    for(int token_idx = 0; token_idx < num_tokens; token_idx++)
    {
        for(int col = 0; col < hidden_size; col++)
        {
            float result = 0.0f;

            // 处理共享专家部分
            if(token_idx >= num_tokens_offset &&
               token_idx < num_tokens_offset + num_tokens_per_rank)
            {
                int local_token_idx = token_idx - num_tokens_offset;

                for(int e = 0; e < num_shared_experts_per_rank; e++)
                {
                    int src_idx = (e * num_tokens_per_rank + local_token_idx) * hidden_size + col;
                    result += static_cast<float>(scatter_tokens[src_idx]);
                }
            }

            // 处理非共享专家部分
            for(int i = 0; i < allocated_tokens; i++)
            {
                if(token_offset[i] == token_idx)
                {
                    int tokens_per_k = allocated_tokens / topk;
                    int k            = i / tokens_per_k;

                    int src_idx =
                        (num_shared_experts_per_rank * num_tokens_per_rank + i) * hidden_size + col;
                    float token_val = static_cast<float>(scatter_tokens[src_idx]);
                    float weight    = expert_weights[token_idx * topk + k];

                    result += token_val * weight;
                }
            }

            hidden_states[token_idx * hidden_size + col] = static_cast<hip_bfloat16>(result);
        }
    }
}

float random_float(float min = -1.0f, float max = 1.0f)
{
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

bool verify_results(const std::vector<hip_bfloat16>& gpu_results,
                    const std::vector<hip_bfloat16>& cpu_results,
                    float tolerance = 1e-3f)
{
    if(gpu_results.size() != cpu_results.size())
    {
        std::cerr << "Size mismatch: GPU " << gpu_results.size() << " vs CPU " << cpu_results.size()
                  << std::endl;
        return false;
    }

    for(size_t i = 0; i < gpu_results.size(); i++)
    {
        float gpu_val = static_cast<float>(gpu_results[i]);
        float cpu_val = static_cast<float>(cpu_results[i]);
        float diff    = fabs(gpu_val - cpu_val);

        if(diff > tolerance)
        {
            std::cerr << "Mismatch at index " << i << ": GPU=" << gpu_val << ", CPU=" << cpu_val
                      << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // 设置随机种子
    srand(42);

    // 参数配置
    const int world_size                  = 4;
    const int num_tokens_per_rank         = 100;
    const int current_rank                = 1;
    const int num_tokens_offset           = num_tokens_per_rank * current_rank;
    const int allocated_tokens            = 200; // 非共享专家token数
    const int hidden_size                 = 1024;
    const int topk                        = 2;
    const int num_shared_experts_per_rank = 2;

    // 总token数
    const int num_tokens = world_size * num_tokens_per_rank;

    // scatter_tokens大小 = (共享专家数 * 当前rank的token数 + 分配的非共享token数)
    // * hidden_size
    const int scatter_tokens_size =
        (num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens) * hidden_size;

    // 1. 创建主机端数据
    std::vector<hip_bfloat16> h_scatter_tokens(scatter_tokens_size);
    std::vector<int> h_token_offset(allocated_tokens);
    std::vector<float> h_expert_weights(num_tokens * topk);
    std::vector<hip_bfloat16> h_hidden_states(num_tokens * hidden_size, hip_bfloat16(0.0f));
    std::vector<hip_bfloat16> h_cpu_results(num_tokens * hidden_size, hip_bfloat16(0.0f));

    // 初始化scatter_tokens
    for(int i = 0; i < scatter_tokens_size; i++)
    {
        h_scatter_tokens[i] = hip_bfloat16(random_float());
    }

    // 初始化token_offset (非共享专家token分配)
    // 当前rank负责的token范围: [num_tokens_offset, num_tokens_offset +
    // num_tokens_per_rank - 1]
    /*
    因为未知每个专家的topk后的token数量，因此对allocated_tokens均分给每个topk。
    实际应用中应该是每个topk拥有数量不等的tokens，所有topk的tokens和为allocated_tokens
    */
    int tokens_per_k = allocated_tokens / topk;
    for(int i = 0; i < allocated_tokens; i++)
    {
        int k                   = i / tokens_per_k;
        int token_index_in_rank = i % num_tokens_per_rank;
        h_token_offset[i]       = num_tokens_offset + token_index_in_rank;
    }

    // 初始化专家权重
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

    // 2. 分配设备内存
    hip_bfloat16* d_scatter_tokens = nullptr;
    int* d_token_offset            = nullptr;
    float* d_expert_weights        = nullptr;
    hip_bfloat16* d_hidden_states  = nullptr;

    HIP_CHECK(hipMalloc(&d_scatter_tokens, scatter_tokens_size * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_token_offset, allocated_tokens * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_weights, num_tokens * topk * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_hidden_states, num_tokens * hidden_size * sizeof(hip_bfloat16)));

    // 3. 复制数据到设备
    HIP_CHECK(hipMemcpy(d_scatter_tokens,
                        h_scatter_tokens.data(),
                        scatter_tokens_size * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_token_offset,
                        h_token_offset.data(),
                        allocated_tokens * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_expert_weights,
                        h_expert_weights.data(),
                        num_tokens * topk * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_hidden_states, 0, num_tokens * hidden_size * sizeof(hip_bfloat16)));

    // 4. 创建HIP流
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // 5. 执行GPU计算
    auto gpu_start = std::chrono::high_resolution_clock::now();
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
    HIP_CHECK(hipStreamSynchronize(stream));
    auto gpu_end = std::chrono::high_resolution_clock::now();

    // 6. 执行CPU计算用于验证
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_moe_gather(h_scatter_tokens,
                   h_token_offset,
                   h_expert_weights,
                   h_cpu_results,
                   num_shared_experts_per_rank,
                   num_tokens_offset,
                   num_tokens_per_rank,
                   allocated_tokens,
                   hidden_size,
                   topk,
                   num_tokens);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // 7. 复制GPU结果回主机
    std::vector<hip_bfloat16> h_gpu_results(num_tokens * hidden_size);
    HIP_CHECK(hipMemcpy(h_gpu_results.data(),
                        d_hidden_states,
                        num_tokens * hidden_size * sizeof(hip_bfloat16),
                        hipMemcpyDeviceToHost));

    // 8. 验证结果
    bool success = verify_results(h_gpu_results, h_cpu_results);

    // 9. 打印性能数据
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);

    std::cout << "\nMoE Gather Results:\n";
    std::cout << "  Token range for current rank: [" << num_tokens_offset << ", "
              << num_tokens_offset + num_tokens_per_rank - 1 << "]\n";
    std::cout << "  Shared experts: " << num_shared_experts_per_rank << "\n";
    std::cout << "  Allocated non-shared tokens: " << allocated_tokens << "\n";
    std::cout << "  Verification: " << (success ? "PASSED" : "FAILED") << "\n";
    std::cout << "  GPU time: " << gpu_duration.count() << " μs\n";
    std::cout << "  CPU time: " << cpu_duration.count() << " μs\n";
    std::cout << "  Speedup: " << static_cast<float>(cpu_duration.count()) / gpu_duration.count()
              << "x\n";

    // 10. 打印前5个token的前5个元素
    std::cout << "\nFirst 5 tokens (first 5 elements):\n";
    for(int token_idx = num_tokens_offset; token_idx < num_tokens_offset + 5; token_idx++)
    {
        std::cout << "Token " << token_idx << ": [";
        for(int col = 0; col < 5; col++)
        {
            float val = static_cast<float>(h_gpu_results[token_idx * hidden_size + col]);
            std::cout << val;
            if(col < 4)
                std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // 11. 清理资源
    HIP_CHECK(hipFree(d_scatter_tokens));
    HIP_CHECK(hipFree(d_token_offset));
    HIP_CHECK(hipFree(d_expert_weights));
    HIP_CHECK(hipFree(d_hidden_states));
    HIP_CHECK(hipStreamDestroy(stream));

    return success ? 0 : 1;
}
