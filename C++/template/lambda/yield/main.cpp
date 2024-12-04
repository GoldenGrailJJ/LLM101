#include <iostream>
#include <vector>

// ��δ�����ʾ�����ʹ��ģ�庯���� lambda ���ʽ������ͬ���͵����ݡ�
// `fetch_data` ��������һ���ɵ��ö����� lambda ���ʽ��������һϵ�����ݽ��д���
// ���ݴ������ݵ����ͣ�����洢����ͬ�������С�

template <class Func>
void fetch_data(Func const& func) {
    // ���� 0 �� 31 ������
    for (int i = 0; i < 32; i++) {
        func(i);          // ���� func �������� i
        func(i + 0.5f);  // ���� func �������� i + 0.5
    }
}

template <typename T> 
void print_data(const std::vector<T>& data) {
    auto print_lambda = [](size_t index, const T& value) {
        std::cout << "The " << index << " element is " << value << " " << std::endl;
    };

    std::cout << "data content: \n";
    for (size_t i = 0; i < data.size(); ++i) {
        print_lambda(i, data[i]);
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> res_i;   // �洢�������������
    std::vector<float> res_f; // �洢���������������

    // ���� fetch_data������һ�� lambda ���ʽ
    fetch_data([&](auto const& x) {
        // ʹ�� std::decay_t ��ȡ x ��ȥ�����úͳ����������
        using T = std::decay_t<decltype(x)>;

        // ���������жϲ��� x �洢����Ӧ��������
        if constexpr (std::is_same_v<T, int>) {
            res_i.push_back(x); // ��� x �� int ���ͣ��洢�� res_i
        }
        else if constexpr (std::is_same_v<T, float>) {
            res_f.push_back(x); // ��� x �� float ���ͣ��洢�� res_f
        }
        });

    print_data(res_i);
    print_data(res_f);
    return 0; // �������
}