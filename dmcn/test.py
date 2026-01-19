import torch
from torchvision.ops import deform_conv2d
import numpy as np
import os

torch.manual_seed(0)

# === 生成浮点张量 ===
def generate_float_tensor(shape):
    return torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)

def validate_dcn_gradients_nhwc():
    """验证 DCN 的梯度计算 - NHWC版本"""
    print("=== DCN 梯度验证 (NHWC布局) ===")
    
    # 设置参数
    batch_size = 2
    channels = 32
    height = 32
    width = 16
    kernel_h = 3
    kernel_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1
    deformable_group = 4
    group = 1
    out_channels = 8
    
    # 验证参数
    assert channels % group == 0, f"channels ({channels}) must be divisible by group ({group})"
    assert channels % deformable_group == 0, f"channels ({channels}) must be divisible by deformable_group ({deformable_group})"
    assert out_channels % group == 0, f"out_channels ({out_channels}) must be divisible by group ({group})"
    
    print(f"参数: batch={batch_size}, channels={channels}, height={height}, width={width}")
    print(f"卷积核: {kernel_h}x{kernel_w}, 分组: {group}, 可变形分组: {deformable_group}")
    print(f"输出通道: {out_channels}")
    print(f"数据布局: NHWC")
    
    # 创建输入张量 - NHWC布局 [batch, height, width, channels]
    input_tensor = generate_float_tensor((batch_size, height, width, channels)).requires_grad_(True)
    
    # 权重形状: [out_channels, kernel_h, kernel_w, channels//group] - NHWC布局
    weight = generate_float_tensor((out_channels, kernel_h, kernel_w, channels // group)).requires_grad_(True)
    
    # offset 形状: [batch_size, height, width, 2 * kernel_h * kernel_w * deformable_group] - NHWC布局
    offset = generate_float_tensor((batch_size, height, width, 2 * kernel_h * kernel_w * deformable_group)).requires_grad_(True)
    
    # mask 形状: [batch_size, height, width, kernel_h * kernel_w * deformable_group] - NHWC布局
    mask = torch.randint(1, 2, (batch_size, height, width, kernel_h * kernel_w * deformable_group), 
                        device="cuda", dtype=torch.int32).float().requires_grad_(True)
    
    print("\n张量信息 (NHWC布局):")
    print(f"input_tensor: {input_tensor.shape} [batch, height, width, channels]")
    print(f"weight: {weight.shape} [out_channels, kernel_h, kernel_w, channels//group]")
    print(f"offset: {offset.shape} [batch, height, width, 2*kernel_h*kernel_w*deformable_group]")
    print(f"mask: {mask.shape} [batch, height, width, kernel_h*kernel_w*deformable_group]")
    
    # 由于PyTorch deform_conv2d只支持NCHW，我们需要先转换为NCHW进行计算
    print("\n注意: PyTorch deform_conv2d需要NCHW输入，将进行布局转换...")
    
    # 转换为NCHW布局用于PyTorch计算
    input_tensor_nchw = input_tensor.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    offset_nchw = offset.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    mask_nchw = mask.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    
    # 权重也需要转换: [out_channels, kernel_h, kernel_w, in_channels//group] -> [out_channels, in_channels//group, kernel_h, kernel_w]
    weight_nchw = weight.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    
    print("转换后的张量信息 (NCHW布局):")
    print(f"input_tensor_nchw: {input_tensor_nchw.shape}")
    print(f"weight_nchw: {weight_nchw.shape}")
    print(f"offset_nchw: {offset_nchw.shape}")
    print(f"mask_nchw: {mask_nchw.shape}")
    
    # 前向传播 - 使用NCHW布局
    print("\n运行 PyTorch DCN 前向传播 (NCHW布局)...")
    try:
        y_ref = deform_conv2d(
            input=input_tensor_nchw, 
            offset=offset_nchw, 
            weight=weight_nchw, 
            mask=mask_nchw, 
            padding=(pad_h, pad_w), 
            stride=(stride_h, stride_w),
            dilation=(dilation_h, dilation_w)
        )
        print(f"输出形状 (NCHW): {y_ref.shape}")
    except Exception as e:
        print(f"前向传播失败: {e}")
        return None, None
    
    # 创建 grad_output - 保持NCHW布局用于反向传播
    grad_output_nchw = torch.randn_like(y_ref)
    
    # 反向传播计算所有梯度
    print("\n运行 PyTorch DCN 反向传播...")
    y_ref.backward(grad_output_nchw)
    
    # 获取梯度并转换回NHWC布局
    print("\n转换梯度回NHWC布局...")
    
    # 转换offset梯度回NHWC
    if offset_nchw.grad is not None:
        grad_offset_nchw = offset_nchw.grad.clone()
        grad_offset_nhwc = grad_offset_nchw.permute(0, 2, 3, 1).contiguous()
    else:
        grad_offset_nhwc = None
    
    # 转换mask梯度回NHWC
    if mask_nchw.grad is not None:
        grad_mask_nchw = mask_nchw.grad.clone()
        grad_mask_nhwc = grad_mask_nchw.permute(0, 2, 3, 1).contiguous()
    else:
        grad_mask_nhwc = None
    
    # 转换input梯度回NHWC
    if input_tensor_nchw.grad is not None:
        grad_input_nchw = input_tensor_nchw.grad.clone()
        grad_input_nhwc = grad_input_nchw.permute(0, 2, 3, 1).contiguous()
    else:
        grad_input_nhwc = None
    
    # 转换weight梯度回NHWC布局: [out_channels, in_channels//group, kernel_h, kernel_w] -> [out_channels, kernel_h, kernel_w, in_channels//group]
    if weight_nchw.grad is not None:
        grad_weight_nchw = weight_nchw.grad.clone()
        grad_weight_nhwc = grad_weight_nchw.permute(0, 2, 3, 1).contiguous()
    else:
        grad_weight_nhwc = None
    
    # 转换grad_output为NHWC布局用于C++验证
    grad_output_nhwc = grad_output_nchw.permute(0, 2, 3, 1).contiguous()
    
    print(f"梯度转换完成:")
    if grad_offset_nhwc is not None:
        print(f"  grad_offset_nhwc 形状: {grad_offset_nhwc.shape}")
    if grad_mask_nhwc is not None:
        print(f"  grad_mask_nhwc 形状: {grad_mask_nhwc.shape}")
    
    # 保存数据供 C++ 验证 - NHWC布局
    output_dir = "test_data_nhwc"
    os.makedirs(output_dir, exist_ok=True)
    
    def save_tensor_nhwc(tensor, filename):
        if tensor is None:
            print(f"跳过保存 {filename} (tensor is None)")
            return None
        # 保存为二进制文件，保持 NHWC 格式
        array = tensor.detach().cpu().numpy().astype(np.float32)
        file_path = os.path.join(output_dir, filename)
        array.tofile(file_path)
        print(f"保存: {filename}, 形状: {array.shape} (NHWC)")
        return array.shape
    
    # 保存输入数据 - NHWC布局
    print("\n保存输入数据 (NHWC布局)...")
    input_shape = save_tensor_nhwc(input_tensor, "input.bin")
    weight_shape = save_tensor_nhwc(weight, "weight.bin")
    offset_shape = save_tensor_nhwc(offset, "offset.bin")
    mask_shape = save_tensor_nhwc(mask, "mask.bin")
    grad_output_shape = save_tensor_nhwc(grad_output_nhwc, "grad_output.bin")
    
    # 保存 PyTorch 的参考梯度 - NHWC布局
    print("\n保存参考梯度 (NHWC布局)...")
    save_tensor_nhwc(grad_offset_nhwc, "grad_offset_pytorch.bin")
    save_tensor_nhwc(grad_mask_nhwc, "grad_mask_pytorch.bin")
    save_tensor_nhwc(grad_weight_nhwc, "grad_weight_pytorch.bin")
    save_tensor_nhwc(grad_input_nhwc, "grad_input_pytorch.bin")
    
    # 保存参数信息
    with open(os.path.join(output_dir, "params.txt"), "w") as f:
        f.write(f"batch_size={batch_size}\n")
        f.write(f"channels={channels}\n")
        f.write(f"height={height}\n")
        f.write(f"width={width}\n")
        f.write(f"kernel_h={kernel_h}\n")
        f.write(f"kernel_w={kernel_w}\n")
        f.write(f"pad_h={pad_h}\n")
        f.write(f"pad_w={pad_w}\n")
        f.write(f"stride_h={stride_h}\n")
        f.write(f"stride_w={stride_w}\n")
        f.write(f"dilation_h={dilation_h}\n")
        f.write(f"dilation_w={dilation_w}\n")
        f.write(f"deformable_group={deformable_group}\n")
        f.write(f"group={group}\n")
        f.write(f"out_channels={out_channels}\n")
    
    print(f"\n所有NHWC数据已保存到 '{output_dir}' 目录")
    
    return grad_offset_nhwc, grad_mask_nhwc

def compare_results_nhwc():
    """比较 PyTorch 和 C++ 的结果 - NHWC版本"""
    print("\n=== 比较 PyTorch 和 C++ 结果 (NHWC布局) ===")
    
    def load_tensor_nhwc(filename):
        file_path = os.path.join("test_data_nhwc", filename)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        # 从参数文件获取形状信息
        params = {}
        with open(os.path.join("test_data_nhwc", "params.txt"), "r") as f:
            for line in f:
                key, value = line.strip().split('=')
                params[key] = int(value)
        
        # 根据文件名确定形状 - NHWC布局
        if filename == "input.bin":
            shape = (params['batch_size'], params['height'], params['width'], params['channels'])
        elif filename == "weight.bin":
            shape = (params['out_channels'], params['kernel_h'], params['kernel_w'], params['channels'] // params['group'])
        elif filename == "offset.bin":
            shape = (params['batch_size'], params['height'], params['width'], 
                    2 * params['kernel_h'] * params['kernel_w'] * params['deformable_group'])
        elif filename == "mask.bin":
            shape = (params['batch_size'], params['height'], params['width'], 
                    params['kernel_h'] * params['kernel_w'] * params['deformable_group'])
        elif filename == "grad_output.bin":
            shape = (params['batch_size'], params['height'], params['width'], params['out_channels'])
        elif "grad_offset" in filename:
            shape = (params['batch_size'], params['height'], params['width'], 
                    2 * params['kernel_h'] * params['kernel_w'] * params['deformable_group'])
        elif "grad_mask" in filename:
            shape = (params['batch_size'], params['height'], params['width'], 
                    params['kernel_h'] * params['kernel_w'] * params['deformable_group'])
        else:
            print(f"未知文件: {filename}")
            return None
        
        # 加载数据
        data = np.fromfile(file_path, dtype=np.float32)
        data = data.reshape(shape)
        print(f"加载: {filename}, 形状: {shape} (NHWC)")
        return torch.from_numpy(data)
    
    # 加载 PyTorch 结果
    grad_offset_pytorch = load_tensor_nhwc("grad_offset_pytorch.bin")
    grad_mask_pytorch = load_tensor_nhwc("grad_mask_pytorch.bin")
    
    # 加载 C++ 结果
    grad_offset_cpp = load_tensor_nhwc("grad_offset_cpp.bin")
    grad_mask_cpp = load_tensor_nhwc("grad_mask_cpp.bin")
    
    if grad_offset_pytorch is None or grad_offset_cpp is None:
        print("无法加载结果文件")
        return False
    
    # 比较结果
    print(f"\n梯度比较 (NHWC布局):")
    
    # grad_offset 比较
    offset_diff = (grad_offset_pytorch - grad_offset_cpp).abs()
    print(f"grad_offset:")
    print(f"  最大绝对误差: {offset_diff.max().item():.6e}")
    print(f"  平均绝对误差: {offset_diff.mean().item():.6e}")
    print(f"  相对误差: {offset_diff.norm() / grad_offset_pytorch.norm():.6e}")
    
    # grad_mask 比较
    mask_diff = (grad_mask_pytorch - grad_mask_cpp).abs()
    print(f"grad_mask:")
    print(f"  最大绝对误差: {mask_diff.max().item():.6e}")
    print(f"  平均绝对误差: {mask_diff.mean().item():.6e}")
    print(f"  相对误差: {mask_diff.norm() / grad_mask_pytorch.norm():.6e}")
    
    # 检查是否匹配
    tolerance = 1e-4
    offset_match = offset_diff.max() < tolerance
    mask_match = mask_diff.max() < tolerance
    
    if offset_match and mask_match:
        print(f"\n✅ 成功: 所有梯度在容差 {tolerance} 内匹配!")
        # 显示前几个匹配的值
        print(f"\n前5个匹配的 grad_offset 值:")
        indices = torch.topk(offset_diff.view(-1), 5).indices
        for i, idx in enumerate(indices):
            idx = idx.item()
            print(f"  [{idx}] PyTorch: {grad_offset_pytorch.view(-1)[idx]:.6f}, "
                  f"C++: {grad_offset_cpp.view(-1)[idx]:.6f}, "
                  f"误差: {offset_diff.view(-1)[idx]:.6e}")
    else:
        print(f"\n❌ 失败: 梯度超过容差 {tolerance}!")
        
        # 显示前几个不匹配的值
        print(f"\n前5个不匹配的 grad_offset 值:")
        indices = torch.topk(offset_diff.view(-1), 5).indices
        for i, idx in enumerate(indices):
            idx = idx.item()
            print(f"  [{idx}] PyTorch: {grad_offset_pytorch.view(-1)[idx]:.6f}, "
                  f"C++: {grad_offset_cpp.view(-1)[idx]:.6f}, "
                  f"误差: {offset_diff.view(-1)[idx]:.6e}")
    
    return offset_match and mask_match

def debug_deform_conv_nhwc():
    """调试 deform_conv2d 函数 - NHWC版本"""
    print("\n=== 调试 deform_conv2d (NHWC布局) ===")
    
    # 使用更简单的参数
    batch_size = 1
    channels = 4
    height = 8
    width = 8
    kernel_h = 3
    kernel_w = 3
    group = 1
    deformable_group = 1
    out_channels = 4
    
    print(f"调试参数: batch={batch_size}, channels={channels}, out_channels={out_channels}")
    print(f"group={group}, deformable_group={deformable_group}")
    print(f"数据布局: NHWC")
    
    # NHWC布局
    input_tensor = torch.randn(batch_size, height, width, channels, device="cuda").requires_grad_(True)
    weight = torch.randn(out_channels, kernel_h, kernel_w, channels // group, device="cuda").requires_grad_(True)
    offset = torch.randn(batch_size, height, width, 2 * kernel_h * kernel_w * deformable_group, device="cuda").requires_grad_(True)
    mask = torch.randn(batch_size, height, width, kernel_h * kernel_w * deformable_group, device="cuda").sigmoid().requires_grad_(True)
    
    print(f"input (NHWC): {input_tensor.shape}")
    print(f"weight (NHWC): {weight.shape}")
    print(f"offset (NHWC): {offset.shape}")
    print(f"mask (NHWC): {mask.shape}")
    
    # 转换为NCHW用于PyTorch deform_conv2d
    input_tensor_nchw = input_tensor.permute(0, 3, 1, 2).contiguous()
    weight_nchw = weight.permute(0, 3, 1, 2).contiguous()
    offset_nchw = offset.permute(0, 3, 1, 2).contiguous()
    mask_nchw = mask.permute(0, 3, 1, 2).contiguous()
    
    print(f"转换后 (NCHW):")
    print(f"input_nchw: {input_tensor_nchw.shape}")
    print(f"weight_nchw: {weight_nchw.shape}")
    print(f"offset_nchw: {offset_nchw.shape}")
    print(f"mask_nchw: {mask_nchw.shape}")
    
    try:
        output = deform_conv2d(input_tensor_nchw, offset_nchw, weight_nchw, mask=mask_nchw, padding=1, stride=1)
        print(f"输出形状 (NCHW): {output.shape}")
        
        # 反向传播
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        print("调试成功!")
        return True
    except Exception as e:
        print(f"调试失败: {e}")
        return False

if __name__ == "__main__":
    print("DCNv2 坐标梯度验证 - NHWC布局")
    print("=" * 60)
    
    # 先运行调试
    print("0. 调试 deform_conv2d (NHWC布局)...")
    debug_success = debug_deform_conv_nhwc()
    
    if debug_success:
        print("\n调试成功，继续主验证...")
    else:
        print("\n调试失败，请检查环境")
        exit(1)
    
    # 步骤 1: 生成测试数据和 PyTorch 参考结果
    if not os.path.exists("test_data_nhwc/input.bin"):
        print("\n1. 生成测试数据 (NHWC布局)...")
        grad_offset, grad_mask = validate_dcn_gradients_nhwc()
        
        if grad_offset is None or grad_mask is None:
            print("生成测试数据失败")
            exit(1)
    else:
        print("1. 测试数据已存在")
    
    print("\n2. 请运行你的 C++ 代码计算 grad_offset 和 grad_mask")
    print("   确保 C++ 代码读取 test_data_nhwc/ 下的输入数据")
    print("   并将结果保存为 test_data_nhwc/grad_offset_cpp.bin 和 test_data_nhwc/grad_mask_cpp.bin")
    
    if os.path.exists("test_data_nhwc/grad_offset_cpp.bin"):
        input("按 Enter 键继续比较结果...")
        
        # 步骤 2: 比较结果
        print("\n3. 比较结果 (NHWC布局)...")
        success = compare_results_nhwc()
        
        if success:
            print("\n DCN NHWC实现与 PyTorch 一致。")
        else:
            print("\n 存在差异。")
    else:
        print("\n未找到 C++ 结果文件，请先运行 C++ 代码")
