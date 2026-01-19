#include <iostream>
#include <iomanip>
#include <cmath>
#include <cfenv> // 用于浮点异常检测（可选）

// 检查编译器是否支持 __fp16
#ifndef __FP16_DECLARED
#ifdef __clang__
// Clang 通常支持 __fp16
#define FP16_AVAILABLE
#elif defined(__GNUC__) && (__GNUC__ >= 12)
// GCC 12+ 支持 __fp16（需开启 -mfp16-format=ieee）
#define FP16_AVAILABLE
#endif
#endif

int main()
{
    float f_val = 65505.0f;
    std::cout << "Original float value: " << std::fixed << std::setprecision(1) << f_val
              << std::endl;

#ifdef FP16_AVAILABLE
    // 启用浮点异常检测（可选，用于观察溢出）
    std::feclearexcept(FE_ALL_EXCEPT);

    // 转换 float -> __fp16
    __fp16 fp16_val = static_cast<__fp16>(f_val);

    // 检查是否发生溢出
    if(std::fetestexcept(FE_OVERFLOW))
    {
        std::cout << "[Warning] Floating-point overflow occurred during conversion!" << std::endl;
    }

    // 转换回 float 以便打印（因为直接打印 __fp16 可能不被支持）
    float f_converted_back = static_cast<float>(fp16_val);

    std::cout << "Converted to FP16, then back to float: " << std::fixed << std::setprecision(1)
              << f_converted_back << std::endl;

    // 直接尝试打印 __fp16（部分编译器支持，但不推荐）
    std::cout << "Direct __fp16 value (if supported): ";
    std::cout << fp16_val << std::endl; // 可能输出乱码或报错，取决于编译器/库

    // 手动解析 FP16 位模式（推荐方式）
    union
    {
        __fp16 val;
        unsigned short bits;
    } fp16_union;
    fp16_union.val = fp16_val;

    unsigned short sign     = (fp16_union.bits >> 15) & 0x1;
    unsigned short exponent = (fp16_union.bits >> 10) & 0x1F;
    unsigned short mantissa = fp16_union.bits & 0x3FF;

    std::cout << "\n--- FP16 Bit Representation ---" << std::endl;
    std::cout << "Sign bit    : " << sign << std::endl;
    std::cout << "Exponent    : " << exponent << " (biased), " << (exponent - 15) << " (unbiased)"
              << std::endl;
    std::cout << "Mantissa    : " << std::hex << "0x" << mantissa << std::dec << std::endl;
    std::cout << "Raw 16-bit  : 0x" << std::hex << fp16_union.bits << std::dec << std::endl;

    // 解释结果
    if(exponent == 0x1F && mantissa == 0)
    {
        std::cout << "==> This represents +Infinity in FP16." << std::endl;
    }
    else if(exponent == 0x1F && mantissa != 0)
    {
        std::cout << "==> This represents NaN in FP16." << std::endl;
    }
    else if(exponent == 0x1E)
    {                                                          // 最大正规数指数
        float max_normal = (2.0f - 1.0f / 1024.0f) * 65536.0f; // (2 - 2^-10) * 2^16 = 65504.0
        std::cout << "==> Maximum representable normal number in FP16 is " << max_normal
                  << std::endl;
    }

#else
    std::cout << ">>> Compiler does not support __fp16 type. <<<" << std::endl;
    std::cout << "Please use Clang or GCC >= 12 with appropriate flags." << std::endl;
#endif

    return 0;
}
