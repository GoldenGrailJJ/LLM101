#include <iostream>
#include <vector>

// 这段代码演示了如何使用 lambda 表达式在一个整数向量中查找特定值的索引，并计算从索引 0 到该索引的元素和。

int main() {
    // 初始化一个整数向量 arr，包含一些整数
    std::vector<int> arr = { 1, 4, 2, 8, 5, 7 };

    // 要查找的目标值
    int tofind = 5;

    // 使用 lambda 表达式查找目标值的索引
    int index = [&] {
        // 遍历向量 arr
        for (int i = 0; i < arr.size(); ++i) {
            // 如果找到目标值，返回其索引
            if (arr[i] == tofind)
                return i;
        }
        // 如果未找到目标值，返回 -1
        return -1;
    }(); // 立即调用 lambda 表达式

    // 输出找到的索引
    std::cout << index << std::endl; // 如果找到，输出索引；否则输出 -1

    // 使用 lambda 表达式计算从索引 0 到找到的索引的元素和
    int sum = [&] {
        int ls = 0; // 初始化和为 0
        // 遍历向量 arr
        for (size_t i = 0; i < arr.size(); ++i) {
            ls += arr[i]; // 累加每个元素
        }
        return ls; // 返回总和
    }(); // 立即调用 lambda 表达式

    // 输出从索引 0 到找到的索引的元素和
    std::cout << "sum from index 0 to " << index << " is " << sum << std::endl;

    return 0; // 程序结束
}