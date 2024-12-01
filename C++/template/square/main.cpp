#include <iostream>
#include <vector>  // ���� std::vector ��ͷ�ļ�

// ģ�庯��������һ������ T �Ĳ����������� T * 3
template <class T>
T three(T t) {
    return t * 3;  // ���� t ������
}

// �ػ��汾������ std::string ����
std::string three(std::string t) {
    return t + t + t;  // �����ַ��� t ��������ƴ�ӣ�
}

//_____________________
// ģ�庯��������һ������ N ��һ������ T1 �Ĳ��� msg
template <int N, class T1>
void show_times(T1 msg) {
    for (int i = 0; i < N; ++i) {  // ѭ�� N ��
        std::cout << msg << std::endl;  // ��� msg
    }
}

//_____________________
// ģ�庯��������һ�� vector ������Ԫ�صĺ�
template <class T2>
T2 sum(std::vector<T2> const& arr) {
    T2 res = 0;  // ��ʼ�����Ϊ 0
    for (int i = 0; i < arr.size(); ++i) {  // ��������
        res += arr[i];  // �ۼ�ÿ��Ԫ��
    }
    return res;  // �����ܺ�
}

//_____________________
// ģ�庯��������� 1 �� n �ĺͣ�֧�ֵ������
template <bool debug>
int sumto(int n) {
    int res = 0;  // ��ʼ�����Ϊ 0
    for (int i = 1; i <= n; ++i) {  // �� 1 �� n ѭ��
        res += i;  // �ۼӵ�ǰֵ
        if constexpr (debug) {  // ��� debug Ϊ true�����������Ϣ
            std::cout << i << "-th: " << res << std::endl;
        }
    }
    return res;  // �����ܺ�
}

// ��������������֧����� vector
template <class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& a) {
    os << "{";
    for (size_t i = 0; i < a.size(); ++i) {  // ���� vector
        os << a[i];  // �����ǰԪ��
        if (i != a.size() - 1) {  // ����������һ��Ԫ��
            os << ", ";  // ������ŷָ�
        }
    }
    os << "}";  // ����������
    return os;  // ���������
}

//_____________________

int main() {
    // ���� three ����ģ��
    std::cout << three<int>(11) << std::endl;  // ��� 33
    std::cout << three<float>(3.14f) << std::endl;  // ��� 9.42
    std::cout << three<double>(2.01012) << std::endl;  // ��� 6.03036

    // �Զ��Ƶ���������
    std::cout << three(11) << std::endl;  // ��� 33
    std::cout << three(3.14f) << std::endl;  // ��� 9.42
    std::cout << three(2.01012) << std::endl;  // ��� 6.03036
    std::cout << three(std::string("hello")) << std::endl;  // ��� "hellohellohello"

    // ���� show_times ����ģ��
    show_times<1>("one");  // ��� "one"
    show_times<3>(42);  // ��� 42 ����

    // ���� sum ����ģ��
    std::vector<int> a = { 4, 3, 2, 1, 0 };
    std::cout << sum(a) << std::endl;  // ��� 10

    // ���� sumto ����ģ��
    std::cout << sumto<true>(4) << std::endl;  // ���������Ϣ������ 10
    std::cout << sumto<false>(4) << std::endl;  // ���� 10���������������Ϣ

    // ����������������
    std::vector<int> a1 = { 1, 4, 2, 8, 5, 7 };
    std::cout << a1 << std::endl;  // ��� {1, 4, 2, 8, 5, 7}
    std::vector<double> b1 = { 3.14, 2.718, 0.618 };
    std::cout << b1 << std::endl;  // ��� {3.14, 2.718, 0.618}

    return 0;
}