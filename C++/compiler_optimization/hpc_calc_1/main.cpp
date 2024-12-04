#include <cstdio>      // C ��׼���������
#include <cstdlib>     // C ��׼�⣬���� rand() ����
#include <vector>      // STL ������
#include <chrono>      // ʱ�������
#include <cmath>       // ��ѧ�⣬���� sqrt() ����
#include <omp.h>       // OpenMP �⣬���ڲ��м���

// ����һ����Χ�� [-1, 1] ֮������������
float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}

// ����ʱȷ������������ʱ����
constexpr int length = 48; // ���峣�� length����ʾ���������

// ÿ��ʵ�����뵽 32 �ֽ�
// ����ʹ�� SIMD ����ʸ����
struct alignas(32) Star {
    float px[length], py[length], pz[length]; // λ������
    float vx[length], vy[length], vz[length]; // �ٶ�����
    float mass[length]; // ����
};

Star stars; // ����һ�� Star ���͵�ʵ�������ڴ洢�����������Ϣ

// ��ʼ����������λ�á��ٶȺ�����
void init() {
    // ѭ��չ����һ��ִ���ĸ�����
    #pragma GCC unroll 4 // �������ܣ�����������չ��ѭ��������ѭ�����ƿ���
    for (size_t i = 0; i < length; i++) {
        stars.px[i] = frand(); // �����ʼ��λ��
        stars.py[i] = frand();
        stars.pz[i] = frand();
        stars.vx[i] = frand(); // �����ʼ���ٶ�
        stars.vy[i] = frand();
        stars.vz[i] = frand();
        stars.mass[i] = frand() + 1; // �����ʼ��������ȷ���������� 1
    }
}

float G = 0.001; // ��������
float eps = 0.001; // ���������ĳ���
float eps_square = eps * eps; // Ԥ���� eps ��ƽ������������
float dt = 0.01; // ʱ�䲽��
float gdt = G * dt; // ���� G * dt����������

// ���������λ�ú��ٶ�
void step() {
    #pragma GCC unroll 4 // �������ܣ�����������չ��ѭ��
    for (size_t i = 0; i < length; ++i) {
        float px = stars.px[i], py = stars.py[i], pz = stars.pz[i]; // ����λ�ã���������
        float vx_plus = 0, vy_plus = 0, vz_plus = 0; // ��ʼ�������ٶ�
        #pragma GCC unroll 4 // �������ܣ�����������չ��ѭ��
        for (size_t j = 0; j < length; ++j) {
            float dx = stars.px[j] - px; // ���� x ����ľ���
            float dy = stars.py[j] - py; // ���� y ����ľ���
            float dz = stars.pz[j] - pz; // ���� z ����ľ���
            float d2 = dx * dx + dy * dy + dz * dz + eps_square; // ��������ƽ��
            d2 *= std::sqrt(d2); // �������������������ظ�����
            float value = stars.mass[j] * gdt / d2; // ��������Ӱ��
            vx_plus += dx * value; // ���� x ������ٶ�����
            vy_plus += dy * value; // ���� y ������ٶ�����
            vz_plus += dz * value; // ���� z ������ٶ�����
        }
        stars.vx[i] += vx_plus; // ����������ٶ�
        stars.vy[i] += vy_plus;
        stars.vz[i] += vz_plus;
    }
    #pragma GCC unroll 4 // �������ܣ�����������չ��ѭ��
    for (size_t i = 0; i < length; ++i) {
        stars.px[i] += stars.vx[i] * dt; // ����λ��
        stars.py[i] += stars.vy[i] * dt;
        stars.pz[i] += stars.vz[i] * dt;
    }
}

// ����ϵͳ��������
float calc() {
    float energy = 0; // ��ʼ������
    for (size_t i = 0; i < length; ++i) {
        float v2 = stars.vx[i] * stars.vx[i] + stars.vy[i] * stars.vy[i] + stars.vz[i] * stars.vz[i]; // �����ٶȵ�ƽ��
        energy += stars.mass[i] * v2 / 2; // ���㶯��
        for (size_t j = 0; j < length; ++j) {
            float dx = stars.px[j] - stars.px[i]; // ����λ�ò�
            float dy = stars.py[j] - stars.py[i];
            float dz = stars.pz[j] - stars.pz[i];
            float d2 = dx * dx + dy * dy + dz * dz + eps_square; // ��������ƽ��
            energy -= stars.mass[j] * stars.mass[i] * G / std::sqrt(d2) / 2; // ��������
        }
    }
    return energy; // ����������
}

// ��׼���Ժ�������������ִ��ʱ��
template <class Func>
long benchmark(Func const& func) {
    auto t0 = std::chrono::steady_clock::now(); // ��ȡ��ʼʱ��
    func(); // ִ�д���ĺ���
    auto t1 = std::chrono::steady_clock::now(); // ��ȡ����ʱ��
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0); // ����ʱ���
    return dt.count(); // ����ʱ�����룩
}

int main() {
    init(); // ��ʼ������
    printf("Initial energy: %f\n", calc()); // �����ʼ����
    auto dt = benchmark([&] { // ��׼���� step ������ִ��ʱ��
        #pragma GCC unroll 64 // �������ܣ�����������չ��ѭ��
        for (int i = 0; i < 100000; i++)
            step(); // ִ�� 100000 �β���
        });
    printf("Final energy: %f\n", calc()); // �����������
    printf("Time elapsed: %ld ms\n", dt); // ���ִ��ʱ��
    return 0; // ���� 0����ʾ������������
}