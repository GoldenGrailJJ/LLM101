/* 基于智能指针实现双向链表 */
#include <cstdio>
#include <memory>

// 定义节点结构体
struct Node {
    // 这个节点包含一个智能指针指向下一个节点，和一个原始指针指向前一个节点
    std::unique_ptr<Node> next;  // 使用 unique_ptr 管理 next 指针
    Node* prev;  // prev 是普通指针，指向前一个节点，暂时没有用 unique_ptr 包装

    int value;  // 存储节点的值

    // 构造函数，初始化节点的值和指针
    explicit Node(int value) : next(nullptr), prev(nullptr), value(value) {
    }

    // 插入新节点，在当前节点后插入一个新节点
    void insert(int value) {
        // 创建一个新节点
        auto node = std::make_unique<Node>(value);
        // 将新节点的 next 指针指向当前节点的 next
        node->next = std::move(next);

        // 如果新节点的 next 不为空，调整相应的 prev 指针
        if (node->next) {
            node->prev = node->next->prev;
            node->next->prev = node.get();
        }

        // 将当前节点的 next 指向新创建的节点
        next = std::move(node);
    }

    // 删除当前节点
    void erase() {
        // 如果当前节点有 next，更新下一个节点的 prev 指针
        if (next) {
            next->prev = prev;
        }
        // 如果当前节点有 prev，更新前一个节点的 next 指针
        if (prev) {
            prev->next = std::move(next);
        }
    }

    // 析构函数，用于打印销毁信息
    ~Node() {
        printf("~Node()\n");    // 每次节点被销毁时会输出一次
    }
};

// 定义链表结构体
struct List {
    std::unique_ptr<Node> head;  // 使用 unique_ptr 管理头节点

    // 默认构造函数
    List() = default;

    // 拷贝构造函数，执行深拷贝
    List(List const& other) {
        printf("List  被拷贝\n");
        head.reset();  // 先重置 head，防止内存泄漏

        // 如果其他链表的头为空，直接返回
        if (!other.front()) return;

        // 获取其他链表的头节点，并初始化新链表的头节点
        Node* pOtherNodeFront = other.front();
        head = std::unique_ptr<Node>(new Node(pOtherNodeFront->value));
        Node* pthisNodeFront = head.get();

        // 遍历其他链表，将每个节点插入到当前链表中
        while (pOtherNodeFront->next) {
            pthisNodeFront->insert(pOtherNodeFront->next->value);
            pthisNodeFront = pthisNodeFront->next.get();
            pOtherNodeFront = pOtherNodeFront->next.get();
        }
    }

    // 删除拷贝赋值函数，防止错误的拷贝赋值操作
    List& operator=(List const&) = delete;

    // 移动构造函数，允许资源的转移
    List(List&&) = default;
    List& operator=(List&&) = default;

    // 获取链表的头节点
    Node* front() const {
        return head.get();
    }

    // 从头部弹出一个元素
    int pop_front() {
        int ret = head->value;  // 获取头节点的值
        head = std::move(head->next);  // 移动头节点，删除它
        return ret;  // 返回原头节点的值
    }

    // 向链表头部插入一个新节点
    void push_front(int value) {
        auto node = std::make_unique<Node>(value);  // 创建新节点
        node->next = std::move(head);  // 将新节点的 next 指向原来的头节点

        // 如果原来有头节点，更新头节点的 prev 指针
        if (node->next) {
            node->next->prev = node.get();
        }

        // 更新头节点为新插入的节点
        head = std::move(node);
    }

    // 获取指定索引位置的节点
    Node* at(size_t index) const {
        auto curr = front();  // 从头节点开始遍历
        for (size_t i = 0; i < index; i++) {
            // 如果越界，打印错误信息并返回 NULL
            if (!curr) {
                printf("Error:Exceed List Volumn!/n");
                return NULL;
            }
            curr = curr->next.get();  // 移动到下一个节点
        }
        return curr;  // 返回指定位置的节点
    }
};

// 打印链表中的元素
void print(List const& lst) {
    printf("[");
    // 从头节点开始遍历链表，打印每个节点的值
    for (auto curr = lst.front(); curr; curr = curr->next.get()) {
        printf(" %d", curr->value);
    }
    printf(" ]\n");
}

// 主函数，演示如何使用链表
int main() {
    List a;  // 创建一个链表

    // 向链表中插入元素
    a.push_front(7);
    a.push_front(5);
    a.push_front(8);
    a.push_front(2);
    a.push_front(9);
    a.push_front(4);
    a.push_front(1);

    print(a);   // 打印链表: [ 1 4 9 2 8 5 7 ]

    // 删除指定位置的节点
    a.at(2)->erase();

    print(a);   // 打印链表: [ 1 4 2 8 5 7 ]

    List b = a;  // 使用深拷贝构造函数创建链表 b

    // 再次删除节点
    a.at(3)->erase();

    print(a);   // 打印链表: [ 1 4 2 5 7 ]
    print(b);   // 打印链表: [ 1 4 2 8 5 7 ]

    // 在链表 a 中插入新节点
    a.at(2)->insert(999);

    print(a);   // 打印链表: [ 1 4 2 999 5 7 ]

    b = {};  // 清空链表 b
    a = {};  // 清空链表 a

    return 0;  // 程序结束
}
