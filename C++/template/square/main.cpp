#include <iostream>
#include <vector>  // 包含 std::vector 的头文件

// 模板函数，接受一个类型 T 的参数，并返回 T * 3
template <class T>
T three(T t) {
    return t * 3;  // 返回 t 的三倍
}

// 特化版本，处理 std::string 类型
std::string three(std::string t) {
    return t + t + t;  // 返回字符串 t 的三倍（拼接）
}

//_____________________
// 模板函数，接受一个整数 N 和一个类型 T1 的参数 msg
template <int N, class T1>
void show_times(T1 msg) {
    for (int i = 0; i < N; ++i) {  // 循环 N 次
        std::cout << msg << std::endl;  // 输出 msg
    }
}

//_____________________
// 模板函数，计算一个 vector 中所有元素的和
template <class T2>
T2 sum(std::vector<T2> const& arr) {
    T2 res = 0;  // 初始化结果为 0
    for (int i = 0; i < arr.size(); ++i) {  // 遍历数组
        res += arr[i];  // 累加每个元素
    }
    return res;  // 返回总和
}

//_____________________
// 模板函数，计算从 1 到 n 的和，支持调试输出
template <bool debug>
int sumto(int n) {
    int res = 0;  // 初始化结果为 0
    for (int i = 1; i <= n; ++i) {  // 从 1 到 n 循环
        res += i;  // 累加当前值
        if constexpr (debug) {  // 如果 debug 为 true，输出调试信息
            std::cout << i << "-th: " << res << std::endl;
        }
    }
    return res;  // 返回总和
}

// 重载输出运算符，支持输出 vector
template <class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& a) {
    os << "{";
    for (size_t i = 0; i < a.size(); ++i) {  // 遍历 vector
        os << a[i];  // 输出当前元素
        if (i != a.size() - 1) {  // 如果不是最后一个元素
            os << ", ";  // 输出逗号分隔
        }
    }
    os << "}";  // 结束大括号
    return os;  // 返回输出流
}

//_____________________

int main() {
    // 测试 three 函数模板
    std::cout << three<int>(11) << std::endl;  // 输出 33
    std::cout << three<float>(3.14f) << std::endl;  // 输出 9.42
    std::cout << three<double>(2.01012) << std::endl;  // 输出 6.03036

    // 自动推导参数类型
    std::cout << three(11) << std::endl;  // 输出 33
    std::cout << three(3.14f) << std::endl;  // 输出 9.42
    std::cout << three(2.01012) << std::endl;  // 输出 6.03036
    std::cout << three(std::string("hello")) << std::endl;  // 输出 "hellohellohello"

    // 测试 show_times 函数模板
    show_times<1>("one");  // 输出 "one"
    show_times<3>(42);  // 输出 42 三次

    // 测试 sum 函数模板
    std::vector<int> a = { 4, 3, 2, 1, 0 };
    std::cout << sum(a) << std::endl;  // 输出 10

    // 测试 sumto 函数模板
    std::cout << sumto<true>(4) << std::endl;  // 输出调试信息并返回 10
    std::cout << sumto<false>(4) << std::endl;  // 返回 10，但不输出调试信息

    // 测试输出运算符重载
    std::vector<int> a1 = { 1, 4, 2, 8, 5, 7 };
    std::cout << a1 << std::endl;  // 输出 {1, 4, 2, 8, 5, 7}
    std::vector<double> b1 = { 3.14, 2.718, 0.618 };
    std::cout << b1 << std::endl;  // 输出 {3.14, 2.718, 0.618}

    return 0;
}