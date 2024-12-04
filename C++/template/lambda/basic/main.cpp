#include <iostream>
#include <vector>  // ���� std::vector ��ͷ�ļ�
#include <functional>

void call_twice(std::function<int(int)> const& func) {
    std::cout << func(0) << std::endl;
    std::cout << func(1) << std::endl;
    std::cout << "size of Func: " << sizeof(func) << std::endl;
}

// 5 
// 1.����ָ������������Ϊ�������ݸ������������Ӷ�ʵ�ָ߽׺����ĸ��
// ��ʹ�ú������Խ�������������Ϊ���룬��ǿ�˴���Ŀ������ԡ�
// 2.ͨ������ָ�룬����������ʱѡ��Ҫ���õĺ�����
// ��ʹ�ô�����������Ը��ݲ�ͬ������ѡ��ͬ�ĺ���ִ�С�

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

    //// ���� lambda ���ʽ���ܻᱻ����Ϊָ�룬
    //// ���������ܻ�Ϊ����ı�������ָ�룬ͨ��ָ��Ĵ�СΪ 8 �ֽڡ�
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