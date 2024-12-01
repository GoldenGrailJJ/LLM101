#include <iostream>
#include <vector>
#include <variant>
#include <algorithm>  // std::min 用于求最小值

// 输出运算符重载，用于输出std::vector容器的内容
template<class T>
std::ostream &operator<<(std::ostream &os, std::vector<T> const &a) {
    os << "{";  // 打印开始的花括号
    for (size_t i = 0; i < a.size(); i++) {
        os << a[i];  // 打印每个元素
        if (i != a.size() - 1)  // 如果不是最后一个元素，输出逗号
            os << ", ";
    }
    os << "}";  // 打印结束的花括号
    return os;  // 返回输出流
}

// vector的逐元素加法运算符重载
template <class T1, class T2>
auto operator+(std::vector<T1> const &a, std::vector<T2> const &b) {
    // 使用decltype推断加法结果类型，T0为a[i] + b[i]的类型
    using T0 = decltype(T1{} + T2{});  // 自动推断加法结果类型
    const int n = std::min(a.size(), b.size());  // 取两个vector的最小长度
    std::vector<T0> res(n, 0);  // 创建一个大小为n的结果向量，初始化为0
    for (int i = 0; i < n; i++) {
        res[i] = a[i] + b[i];  // 对应位置元素相加
    }
    return res;  // 返回结果向量
}

// variant类型的加法运算符重载
template <class T1, class T2>
std::variant<T1, T2> operator+(std::variant<T1, T2> const &a, std::variant<T1, T2> const &b) {
    // 使用std::visit来访问variant中的具体类型，并执行加法
    auto lambda_add = [&](auto const& t1, auto const& t2)->std::variant<T1, T2> {
        return t1 + t2;
    };
    return std::visit(lambda_add, a, b);
}

// variant类型和普通类型的加法运算符重载
template <class T1, class T2>
auto operator+(std::variant<T1, T2> const &a, T2 const &b) {
    // 将T2转换为std::variant<T1, T2>类型，然后递归调用variant的加法运算
    std::variant<T1, T2> b1 = b;
    return a + b1;
};

// 反向操作，普通类型和variant类型的加法
template <class T1, class T2, class T3>
auto operator+(T3 const &a, std::variant<T1, T2> const &b) {
    // 将T3转换为std::variant<T1, T2>类型，然后递归调用variant的加法运算
    std::variant<T1, T2> a1 = a;
    return a1 + b;
}

// 输出variant类型的重载运算符，自动匹配并打印variant中存储的类型
template <class T1, class T2>
std::ostream &operator<<(std::ostream &os, std::variant<T1, T2> const &a) {
    // 使用std::visit访问variant中的元素并输出
    std::visit([&](auto const &it) {
        os << it << std::endl;  // 打印variant中的值
    }, a);
    return os;  // 返回输出流
}

int main() {
    std::vector<int> a = {1, 4, 2, 8, 5, 7};  // 整型vector
    std::cout << a << std::endl;  // 输出vector a
    std::vector<double> b = {3.14, 2.718, 0.618};  // 浮动型vector
    std::cout << b << std::endl;  // 输出vector b

    // 进行vector的加法操作
    auto c = a + b;  // 进行逐元素加法

    // 输出c的类型，应该是std::vector<double>，因为int和double相加会转为double
    std::cout << std::is_same_v<decltype(c), std::vector<double>> << std::endl;  

    // 输出加法结果
    std::cout << c << std::endl;  // 输出 {4.14, 6.718, 2.618}

    // 使用variant存储结果，进行variant加法操作
    std::variant<std::vector<int>, std::vector<double>> d = c;
    std::variant<std::vector<int>, std::vector<double>> e = a;
    d = d + c + e;  // 进行variant加法操作

    // 输出加法后的variant结果
    std::cout << d << std::endl;  // 输出 {9.28, 17.436, 7.236}

    return 0;
}
