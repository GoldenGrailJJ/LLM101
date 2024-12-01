#include <iostream>
#include <vector>

// ��δ�����ʾ�����ʹ�� lambda ���ʽ��һ�����������в����ض�ֵ������������������� 0 ����������Ԫ�غ͡�

int main() {
    // ��ʼ��һ���������� arr������һЩ����
    std::vector<int> arr = { 1, 4, 2, 8, 5, 7 };

    // Ҫ���ҵ�Ŀ��ֵ
    int tofind = 5;

    // ʹ�� lambda ���ʽ����Ŀ��ֵ������
    int index = [&] {
        // �������� arr
        for (int i = 0; i < arr.size(); ++i) {
            // ����ҵ�Ŀ��ֵ������������
            if (arr[i] == tofind)
                return i;
        }
        // ���δ�ҵ�Ŀ��ֵ������ -1
        return -1;
    }(); // �������� lambda ���ʽ

    // ����ҵ�������
    std::cout << index << std::endl; // ����ҵ������������������� -1

    // ʹ�� lambda ���ʽ��������� 0 ���ҵ���������Ԫ�غ�
    int sum = [&] {
        int ls = 0; // ��ʼ����Ϊ 0
        // �������� arr
        for (size_t i = 0; i < arr.size(); ++i) {
            ls += arr[i]; // �ۼ�ÿ��Ԫ��
        }
        return ls; // �����ܺ�
    }(); // �������� lambda ���ʽ

    // ��������� 0 ���ҵ���������Ԫ�غ�
    std::cout << "sum from index 0 to " << index << " is " << sum << std::endl;

    return 0; // �������
}