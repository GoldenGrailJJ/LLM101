#include <iostream>
#include <vector>  // 包含 std::vector 的头文件
#include <functional>

void call_twice(std::function<int(int)> const& func) {
    std::cout << func(0) << std::endl;
    std::cout << func(1) << std::endl;
    std::cout << "size of Func: " << sizeof(func) << std::endl;
}

// 5 
// 1.函数指针允许将函数作为参数传递给其他函数，从而实现高阶函数的概念。
// 这使得函数可以接受其他函数作为输入，增强了代码的可重用性。
// 2.通过函数指针，可以在运行时选择要调用的函数。
// 这使得代码更加灵活，可以根据不同的条件选择不同的函数执行。

std::function<int(int)> make_twice(int fac) {
    return [=](int n) {
        return n * fac;
    };
}

// 1-4
//template <class Func>
//void call_twice(Func const &func) {
//    std::cout << func(0) << std::endl;
//    std::cout << func(1) << std::endl;
//    std::cout << "size of Func: " << sizeof(Func) << std::endl;
//}

// 3
//auto make_twice() {
//    return [](int n) {
//        return n * 2;
//    };
//}

// 4
//auto make_twice(int fac) {
//    return [=](int n) {
//        return n * fac;
//    };
//}

int main() {
    // 1
    //auto myfunc = [](int n) {
    //    printf("Number %d\n", n);
    //};
    //call_twice(myfunc);

    // 2 
    //int fac = 2;
    //int counter = 0;
    //auto twice = [&](int n) {
    //    counter++;
    //    return n * fac;
    //};

    //// 由于 lambda 表达式可能会被捕获为指针，
    //// 编译器可能会为捕获的变量分配指针，通常指针的大小为 8 字节。
    //call_twice(twice);

    //printf("call func %d times", counter);

    // 3
    //auto twice = make_twice();
    //call_twice(twice);

    // 4 
    //auto twice = make_twice(99);
    //call_twice(twice);

    // 5
    auto twice = make_twice(99);
    call_twice(twice);
    
    return 0;

}