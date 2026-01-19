#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// 自定义 bfloat16 类型（简化表示）
struct bfloat16
{
    uint16_t value;
    operator float() const
    {
        uint32_t tmp = value << 16;
        return *(float *)&tmp;
    }
};

void moe_scatter_dynamic_quant(
    // 输入
    const bfloat16 *hidden_states,   // [num_tokens, hidden_size]
    const int32_t *selected_experts, // [num_tokens, topk]
    const float *smooth_scale,       // [num_shared + num_experts, hidden_size]
    int num_tokens,
    int hidden_size,
    int topk,
    int num_shared_experts_per_rank,
    int num_experts_per_rank,

    // 输出
    bfloat16 *scatter_tokens,       // [total_tokens, hidden_size]
    float *scatter_per_token_scale, // [total_tokens]
    int32_t *experts_token_count,   // [num_experts_per_rank]
    int32_t *token_offset,          // [allocated_tokens, topk]

    // 属性
    int expert_start_idx,           // 专家起始索引
    int expert_end_idx,             // 专家结束索引
    int num_tokens_offset,
    int num_tokens_per_rank)
{
    // 验证输入
    if (num_tokens <= 0 || hidden_size <= 0 || topk <= 0)
    {
        throw std::invalid_argument("无效的输入维度");
    }
    if (expert_start_idx < 0 || expert_end_idx <= expert_start_idx)
    {
        throw std::invalid_argument("无效的专家索引范围");
    }
    if (num_shared_experts_per_rank < 0 || num_experts_per_rank < 0)
    {
        throw std::invalid_argument("无效的专家数量");
    }

    // 当前 rank 的总专家数（共享 + 非共享）
    const int total_experts_in_rank = num_shared_experts_per_rank + num_experts_per_rank;

    // 1. 初始化专家 token 计数（仅非共享专家）
    std::memset(experts_token_count, 0, num_experts_per_rank * sizeof(int32_t));

    // 2. 初始化 token_offset 为 -1（无效值）
    const int allocated_tokens = num_tokens * topk;
    std::fill_n(token_offset, allocated_tokens * topk, -1);

    // 3. 初始化专家当前计数器（用于定位非共享专家 token）
    std::vector<int> expert_current_count(num_experts_per_rank, 0);

    // 4. 计算共享专家的全局偏移量
    const int shared_tokens_start = 0;
    const int non_shared_tokens_start = num_shared_experts_per_rank * num_tokens_per_rank;

    // 5. 遍历所有 token
    int total_tokens_written = 0;
    int non_shared_tokens_written = 0;

    for (int token_idx = 0; token_idx < num_tokens; token_idx++)
    {
        // 当前 token 在共享专家中的位置
        const int shared_token_local_idx = token_idx - num_tokens_offset;
        const bool is_shared_token = (shared_token_local_idx >= 0) &&
                                     (shared_token_local_idx < num_tokens_per_rank);

        // 临时存储当前 token 的偏移量（用于非共享 token）
        std::vector<int> current_token_offsets(topk, -1);

        // 遍历当前 token 的所有 topk 专家
        for (int k = 0; k < topk; k++)
        {
            const int expert_idx = selected_experts[token_idx * topk + k];
            const bfloat16 *token_data = hidden_states + token_idx * hidden_size;

            // 处理共享专家
            if (expert_idx < num_shared_experts_per_rank)
            {
                if (is_shared_token)
                {
                    // 计算在输出中的位置
                    const int output_idx = expert_idx * num_tokens_per_rank + shared_token_local_idx;

                    // 复制 token 数据
                    for (int h = 0; h < hidden_size; h++)
                    {
                        scatter_tokens[output_idx * hidden_size + h] = token_data[h];  // 进行散列操作
                    }

                    // 获取专家的量化缩放因子
                    const float *expert_scale = smooth_scale + expert_idx * hidden_size;
                    float max_scale = 0.0f;

                    // 计算 token 的缩放因子（取最大值）
                    for (int h = 0; h < hidden_size; h++)
                    {
                        if (expert_scale[h] > max_scale)
                        {
                            max_scale = expert_scale[h];                              // 进行量化操作
                        }
                    }

                    // 保存缩放因子
                    scatter_per_token_scale[output_idx] = max_scale;

                    // 记录当前 token 的偏移量
                    current_token_offsets[k] = output_idx;

                    total_tokens_written++;
                }
            }
            // 处理当前 rank 的非共享专家
            else if (expert_idx >= expert_start_idx && expert_idx < expert_end_idx)
            {
                // 计算本地专家索引
                const int local_expert_idx = expert_idx - expert_start_idx;
                const int current_count = expert_current_count[local_expert_idx];

                // 计算在输出中的位置
                const int output_idx = non_shared_tokens_start +
                                       experts_token_count[local_expert_idx] +
                                       current_count;

                // 复制 token 数据
                for (int h = 0; h < hidden_size; h++)
                {
                    scatter_tokens[output_idx * hidden_size + h] = token_data[h];   // 散列操作
                }

                // 获取专家的量化缩放因子
                const float *expert_scale = smooth_scale +
                                            (num_shared_experts_per_rank + local_expert_idx) * hidden_size;
                float max_scale = 0.0f;

                // 计算 token 的缩放因子（取最大值）
                for (int h = 0; h < hidden_size; h++)
                {
                    if (expert_scale[h] > max_scale)
                    {
                        max_scale = expert_scale[h];
                    }
                }

                // 保存缩放因子
                scatter_per_token_scale[output_idx] = max_scale;

                // 更新专家计数器
                expert_current_count[local_expert_idx]++;
                experts_token_count[local_expert_idx]++;

                // 记录当前 token 的偏移量
                current_token_offsets[k] = output_idx;

                total_tokens_written++;
                non_shared_tokens_written++;
            }
        }

        // 保存当前 token 的偏移量信息（仅非共享 token）
        if (non_shared_tokens_written > 0)
        {
            for (int k = 0; k < topk; k++)
            {
                const int output_row = non_shared_tokens_written - 1;
                token_offset[output_row * topk + k] = current_token_offsets[k];
            }
        }
    }
}


int main() {
    // 参数设置
    int num_tokens = 1024;
    int hidden_size = 768;
    int topk = 2;
    int num_shared_experts_per_rank = 2;
    int num_experts_per_rank = 4;
    int expert_start_idx = 2;
    int expert_end_idx = 6;
    int num_tokens_offset = 0;
    int num_tokens_per_rank = num_tokens; // 假设每个 rank 处理所有 token
    
    // 计算输出大小
    int shared_tokens_size = num_shared_experts_per_rank * num_tokens_per_rank;
    int allocated_tokens = num_tokens * topk;
    int total_tokens = shared_tokens_size + allocated_tokens;
    
    // 分配输入
    bfloat16* hidden_states = new bfloat16[num_tokens * hidden_size];
    int32_t* selected_experts = new int32_t[num_tokens * topk];
    float* smooth_scale = new float[(num_shared_experts_per_rank + num_experts_per_rank) * hidden_size];
    
    // 分配输出
    bfloat16* scatter_tokens = new bfloat16[total_tokens * hidden_size];
    float* scatter_per_token_scale = new float[total_tokens];
    int32_t* experts_token_count = new int32_t[num_experts_per_rank];
    int32_t* token_offset = new int32_t[allocated_tokens * topk];
    
    // 初始化数据
    // ... (填充 hidden_states, selected_experts, smooth_scale 的实际数据)
    
    try {
        // 调用 scatter 函数
        moe_scatter_dynamic_quant(
            hidden_states, selected_experts, smooth_scale,
            num_tokens, hidden_size, topk,
            num_shared_experts_per_rank, num_experts_per_rank,
            scatter_tokens, scatter_per_token_scale,
            experts_token_count, token_offset,
            expert_start_idx, expert_end_idx,
            num_tokens_offset, num_tokens_per_rank
        );
        
        // 使用处理后的数据...
        // ...
        
    } catch (const std::exception& e) {
        // 错误处理
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    // 释放内存
    delete[] hidden_states;
    delete[] selected_experts;
    delete[] smooth_scale;
    delete[] scatter_tokens;
    delete[] scatter_per_token_scale;
    delete[] experts_token_count;
    delete[] token_offset;
    
    return 0;
}
