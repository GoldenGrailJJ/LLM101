#include <iostream>
#include <vector>
#include <variant>
#include <algorithm>  // std::min ��������Сֵ

// �����������أ��������std::vector����������
template<class T>
std::ostream &operator<<(std::ostream &os, std::vector<T> const &a) {
    os << "{";  // ��ӡ��ʼ�Ļ�����
    for (size_t i = 0; i < a.size(); i++) {
        os << a[i];  // ��ӡÿ��Ԫ��
        if (i != a.size() - 1)  // ����������һ��Ԫ�أ��������
            os << ", ";
    }
    os << "}";  // ��ӡ�����Ļ�����
    return os;  // ���������
}

// vector����Ԫ�ؼӷ����������
template <class T1, class T2>
auto operator+(std::vector<T1> const &a, std::vector<T2> const &b) {
    // ʹ��decltype�ƶϼӷ�������ͣ�T0Ϊa[i] + b[i]������
    using T0 = decltype(T1{} + T2{});  // �Զ��ƶϼӷ��������
    const int n = std::min(a.size(), b.size());  // ȡ����vector����С����
    std::vector<T0> res(n, 0);  // ����һ����СΪn�Ľ����������ʼ��Ϊ0
    for (int i = 0; i < n; i++) {
        res[i] = a[i] + b[i];  // ��Ӧλ��Ԫ�����
    }
    return res;  // ���ؽ������
}

// variant���͵ļӷ����������
template <class T1, class T2>
std::variant<T1, T2> operator+(std::variant<T1, T2> const &a, std::variant<T1, T2> const &b) {
    // ʹ��std::visit������variant�еľ������ͣ���ִ�мӷ�
    auto lambda_add = [&](auto const& t1, auto const& t2)->std::variant<T1, T2> {
        return t1 + t2;
    };
    return std::visit(lambda_add, a, b);
}

// variant���ͺ���ͨ���͵ļӷ����������
template <class T1, class T2>
auto operator+(std::variant<T1, T2> const &a, T2 const &b) {
    // ��T2ת��Ϊstd::variant<T1, T2>���ͣ�Ȼ��ݹ����variant�ļӷ�����
    std::variant<T1, T2> b1 = b;
    return a + b1;
};

// �����������ͨ���ͺ�variant���͵ļӷ�
template <class T1, class T2, class T3>
auto operator+(T3 const &a, std::variant<T1, T2> const &b) {
    // ��T3ת��Ϊstd::variant<T1, T2>���ͣ�Ȼ��ݹ����variant�ļӷ�����
    std::variant<T1, T2> a1 = a;
    return a1 + b;
}

// ���variant���͵�������������Զ�ƥ�䲢��ӡvariant�д洢������
template <class T1, class T2>
std::ostream &operator<<(std::ostream &os, std::variant<T1, T2> const &a) {
    // ʹ��std::visit����variant�е�Ԫ�ز����
    std::visit([&](auto const &it) {
        os << it << std::endl;  // ��ӡvariant�е�ֵ
    }, a);
    return os;  // ���������
}

int main() {
    std::vector<int> a = {1, 4, 2, 8, 5, 7};  // ����vector
    std::cout << a << std::endl;  // ���vector a
    std::vector<double> b = {3.14, 2.718, 0.618};  // ������vector
    std::cout << b << std::endl;  // ���vector b

    // ����vector�ļӷ�����
    auto c = a + b;  // ������Ԫ�ؼӷ�

    // ���c�����ͣ�Ӧ����std::vector<double>����Ϊint��double��ӻ�תΪdouble
    std::cout << std::is_same_v<decltype(c), std::vector<double>> << std::endl;  

    // ����ӷ����
    std::cout << c << std::endl;  // ��� {4.14, 6.718, 2.618}

    // ʹ��variant�洢���������variant�ӷ�����
    std::variant<std::vector<int>, std::vector<double>> d = c;
    std::variant<std::vector<int>, std::vector<double>> e = a;
    d = d + c + e;  // ����variant�ӷ�����

    // ����ӷ����variant���
    std::cout << d << std::endl;  // ��� {9.28, 17.436, 7.236}

    return 0;
}
