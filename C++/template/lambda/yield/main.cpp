#include <iostream>
#include <vector>

// 这段代码演示了如何使用模板函数和 lambda 表达式来处理不同类型的数据。
// `fetch_data` 函数接受一个可调用对象（如 lambda 表达式），并对一系列数据进行处理。
// 根据传入数据的类型，将其存储到不同的向量中。

template <class Func>
void fetch_data(Func const& func) {
    // 遍历 0 到 31 的整数
    for (int i = 0; i < 32; i++) {
        func(i);          // 调用 func 处理整数 i
        func(i + 0.5f);  // 调用 func 处理浮点数 i + 0.5
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
    std::vector<int> res_i;   // 存储整数结果的向量
    std::vector<float> res_f; // 存储浮点数结果的向量

    // 调用 fetch_data，传入一个 lambda 表达式
    fetch_data([&](auto const& x) {
        // 使用 std::decay_t 获取 x 的去掉引用和常量后的类型
        using T = std::decay_t<decltype(x)>;

        // 根据类型判断并将 x 存储到相应的向量中
        if constexpr (std::is_same_v<T, int>) {
            res_i.push_back(x); // 如果 x 是 int 类型，存储到 res_i
        }
        else if constexpr (std::is_same_v<T, float>) {
            res_f.push_back(x); // 如果 x 是 float 类型，存储到 res_f
        }
        });

    print_data(res_i);
    print_data(res_f);
    return 0; // 程序结束
}