#include <cstdio>      // C 标准输入输出库
#include <cstdlib>     // C 标准库，包含 rand() 函数
#include <vector>      // STL 向量库
#include <chrono>      // 时间测量库
#include <cmath>       // 数学库，包含 sqrt() 函数
#include <omp.h>       // OpenMP 库，用于并行计算

// 生成一个范围在 [-1, 1] 之间的随机浮点数
float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}

// 编译时确定，避免运行时开销
constexpr int length = 48; // 定义常量 length，表示星体的数量

// 每个实例对齐到 32 字节
// 可以使用 SIMD 进行矢量化
struct alignas(32) Star {
    float px[length], py[length], pz[length]; // 位置坐标
    float vx[length], vy[length], vz[length]; // 速度坐标
    float mass[length]; // 质量
};

Star stars; // 创建一个 Star 类型的实例，用于存储所有星体的信息

// 初始化星体的随机位置、速度和质量
void init() {
    // 循环展开，一次执行四个操作
    #pragma GCC unroll 4 // 提升性能：编译器尝试展开循环，减少循环控制开销
    for (size_t i = 0; i < length; i++) {
        stars.px[i] = frand(); // 随机初始化位置
        stars.py[i] = frand();
        stars.pz[i] = frand();
        stars.vx[i] = frand(); // 随机初始化速度
        stars.vy[i] = frand();
        stars.vz[i] = frand();
        stars.mass[i] = frand() + 1; // 随机初始化质量，确保质量大于 1
    }
}

float G = 0.001; // 引力常数
float eps = 0.001; // 避免除以零的常数
float eps_square = eps * eps; // 预计算 eps 的平方，提升性能
float dt = 0.01; // 时间步长
float gdt = G * dt; // 计算 G * dt，提升性能

// 更新星体的位置和速度
void step() {
    #pragma GCC unroll 4 // 提升性能：编译器尝试展开循环
    for (size_t i = 0; i < length; ++i) {
        float px = stars.px[i], py = stars.py[i], pz = stars.pz[i]; // 缓存位置，提升性能
        float vx_plus = 0, vy_plus = 0, vz_plus = 0; // 初始化增量速度
        #pragma GCC unroll 4 // 提升性能：编译器尝试展开循环
        for (size_t j = 0; j < length; ++j) {
            float dx = stars.px[j] - px; // 计算 x 方向的距离
            float dy = stars.py[j] - py; // 计算 y 方向的距离
            float dz = stars.pz[j] - pz; // 计算 z 方向的距离
            float d2 = dx * dx + dy * dy + dz * dz + eps_square; // 计算距离的平方
            d2 *= std::sqrt(d2); // 计算距离的立方，避免重复计算
            float value = stars.mass[j] * gdt / d2; // 计算引力影响
            vx_plus += dx * value; // 更新 x 方向的速度增量
            vy_plus += dy * value; // 更新 y 方向的速度增量
            vz_plus += dz * value; // 更新 z 方向的速度增量
        }
        stars.vx[i] += vx_plus; // 更新星体的速度
        stars.vy[i] += vy_plus;
        stars.vz[i] += vz_plus;
    }
    #pragma GCC unroll 4 // 提升性能：编译器尝试展开循环
    for (size_t i = 0; i < length; ++i) {
        stars.px[i] += stars.vx[i] * dt; // 更新位置
        stars.py[i] += stars.vy[i] * dt;
        stars.pz[i] += stars.vz[i] * dt;
    }
}

// 计算系统的总能量
float calc() {
    float energy = 0; // 初始化能量
    for (size_t i = 0; i < length; ++i) {
        float v2 = stars.vx[i] * stars.vx[i] + stars.vy[i] * stars.vy[i] + stars.vz[i] * stars.vz[i]; // 计算速度的平方
        energy += stars.mass[i] * v2 / 2; // 计算动能
        for (size_t j = 0; j < length; ++j) {
            float dx = stars.px[j] - stars.px[i]; // 计算位置差
            float dy = stars.py[j] - stars.py[i];
            float dz = stars.pz[j] - stars.pz[i];
            float d2 = dx * dx + dy * dy + dz * dz + eps_square; // 计算距离的平方
            energy -= stars.mass[j] * stars.mass[i] * G / std::sqrt(d2) / 2; // 计算势能
        }
    }
    return energy; // 返回总能量
}

// 基准测试函数，测量函数执行时间
template <class Func>
long benchmark(Func const& func) {
    auto t0 = std::chrono::steady_clock::now(); // 获取开始时间
    func(); // 执行传入的函数
    auto t1 = std::chrono::steady_clock::now(); // 获取结束时间
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0); // 计算时间差
    return dt.count(); // 返回时间差（毫秒）
}

int main() {
    init(); // 初始化星体
    printf("Initial energy: %f\n", calc()); // 输出初始能量
    auto dt = benchmark([&] { // 基准测试 step 函数的执行时间
        #pragma GCC unroll 64 // 提升性能：编译器尝试展开循环
        for (int i = 0; i < 100000; i++)
            step(); // 执行 100000 次步进
        });
    printf("Final energy: %f\n", calc()); // 输出最终能量
    printf("Time elapsed: %ld ms\n", dt); // 输出执行时间
    return 0; // 返回 0，表示程序正常结束
}