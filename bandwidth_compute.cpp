#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <vector>
#include <iostream>

#define HIP_CHECK(call)                                                 \
    {                                                                   \
        hipError_t err = call;                                          \
        if (err != hipSuccess)                                          \
        {                                                               \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl;  \
            exit(1);                                                    \
        }                                                               \
    }

__global__ void vectorAdd(float *C, const float *A, const float *B, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
        interpret_cast<const volatile float &>(C[idx]); // Prevent compiler optimization
    }
}

void analyzeVectorAdd()
{
    const int n = 1 << 24; // 16M elements
    std::vector<float> h_A(n), h_B(n), h_C(n);

    for (int i = 0; i < n; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(n - 1);
    }

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, n * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), n * sizeof(float), hipMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    vectorAdd<<<gridSize, blockSize>>>(d_C, d_A, d_B, n);
    hipDeviceSynchronize(); // Ensure the kernel execution is complete

    hipEventRecord(start);
    for (int i = 0; i < 100; ++i)
    {
        vectorAdd<<<gridSize, blockSize>>>(d_C, d_A, d_B, n);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop); // Wait for the event to complete

    float elapsedTime = 0.0f;
    hipEventElapsedTime(&elapsedTime, start, stop); // Get elapsed time in milliseconds
    float averageTime = elapsedTime / 100.0f;

    double elementsPerSecond = n / (averageTime * 1e-3);                 // elements per second
    double gigaElementsPerSecond = elementsPerSecond / 1e9;              // giga-elements per second
    double bandwidth = gigaElementsPerSecond * sizeof(float) * 3 * 1e-9; // 3 for A, B, C 1e-9 to convert to GB/s

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    int cuCount = prop.multiProcessorCount;
    int clockRate = prop.clockRate; // kHz
    int coresPerCU = (std::string(prop.gcnArchName).find("gfx90a") != std::string::npos) ? 128 : 64;

    std::cout << "CU count: " << cuCount << std::endl;
    std::cout << "Clock Rate: " << clockRate * 1e-3 << " MHz" << std::endl;  // 时钟频率
    std::cout << "Cores per CU: " << coresPerCU << std::endl;

    double peakTflops = static_cast<double>(cuCount) * coresPerCU * 2 * clockRate * 1e3 * 1e-12;
    double peakBandwidth = prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8 * 2 * 1e-9;  // 8 bits to bytes, 2 for read/write

    printf("%f MHz, %f TB/s\n", prop.memoryClockRate * 1e-3, prop.memoryBusWidth * 1e-3);

    // 计算利用率
    double computeUtil = (gigaElementsPerSecond / (peakTflops * 1000)) * 100;
    double bandwidthUtil = (bandwidth / peakBandwidth) * 100;

    // 打印结果
    std::cout << "===== 向量加法 性能分析 =====" << std::endl;
    std::cout << "数据大小: " << n << " 元素" << std::endl;
    std::cout << "执行时间: " << averageTime << " ms" << std::endl;
    std::cout << "计算性能: " << gigaElementsPerSecond << " gigaElementsPerSecond (峰值 " << peakTflops * 1000 << " gigaElementsPerSecond)" << std::endl;
    std::cout << "内存带宽: " << bandwidth << " GB/s (峰值 " << peakBandwidth << " GB/s)" << std::endl;
    std::cout << "计算利用率: " << computeUtil << "%" << std::endl;
    std::cout << "带宽利用率: " << bandwidthUtil << "%" << std::endl;

    if (bandwidthUtil > 70)
    {
        std::cout << "结论: 访存密集型kernel (接近带宽峰值)" << std::endl;
    }
    else
    {
        std::cout << "结论: 访存密集型kernel (带宽利用率不足)" << std::endl;
    }

    std::cout << "=============================\n"
              << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main(int argc, char const *argv[])
{
    analyzeVectorAdd();
    std::cout << "向量加法性能分析完成。" << std::endl;
    return 0;
}

