#include <iostream>
#include <vector>  // 包含 std::vector 的头文件

template <class T1, class T2>
auto add(std::vector<T1> const& a, std::vector<T2> const& b) {
    using T0 = decltype(T1{} *T2{});
    std::vector<T0> res;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        res.push_back(a[i] + b[i]);
    }
    return res;
}

int main() {
    std::vector<int> a = { 2, 3, 4 };
    std::vector<float> b = { 0.5f, 1.0f, 2.0f };
    auto c = add(a, b);
    for (size_t i = 0; i < c.size(); ++i) {
        std::cout << c[i] << std::endl;
    }
    return 0;

}