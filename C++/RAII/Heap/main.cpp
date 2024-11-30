#include <cstdio>
#include <memory>
#include <vector>
#include <algorithm>

// Define the heap node structure
struct HeapNode {
    int value;  // Value of the node

    explicit HeapNode(int value) : value(value) 
    {
        printf("HeapNode %d created\n", value);
    }

    ~HeapNode() 
    {
        printf("~HeapNode %d destroyed\n", value);
    }
};

struct Heap
{
    std::vector<std::unique_ptr<HeapNode>> nodes;  // Vector to store heap nodes

    Heap() = default;  // Default constructor

    // Insert a new value into the heap
    void insert(int value) {
        auto node = std::make_unique<HeapNode>(value);
        nodes.push_back(std::move(node));
        heapify_up(nodes.size() - 1);  // Maintain heap property
    }

    // Remove and return the top value of the heap
    int pop_top() {
        if (nodes.empty()) {
            printf("Error: Heap is empty!\n");
            return -1;  // Return -1 if the heap is empty
        }

        int top_value = nodes.front()->value;  // Get the top value

        nodes.front() = std::move(nodes.back());  // Move the last node to the top
        nodes.pop_back();  // Remove the last node

        heapify_down(0);  // Maintain heap property

        return top_value;  // Return the top value
    }

    // Get the size of the heap
    size_t size() const {
        return nodes.size();
    }

    // Print the elements of the heap
    void print() const {
        printf("[ ");
        for (const auto& node : nodes) {
            printf("%d ", node->value);
        }
        printf("]\n");
    }

private:
    // Maintain heap property by moving the node up
    void heapify_up(size_t index) {
        size_t parent = (index - 1) / 2;
        if (index > 0 && nodes[index]->value > nodes[parent]->value) {
            std::swap(nodes[index], nodes[parent]);
            heapify_up(parent);
        }
    }

    // Maintain heap property by moving the node down
    void heapify_down(size_t index) {
        size_t left = 2 * index + 1;
        size_t right = 2 * index + 2;
        size_t largest = index;

        if (left < nodes.size() && nodes[left]->value > nodes[largest]->value) {
            largest = left;
        }

        if (right < nodes.size() && nodes[right]->value > nodes[largest]->value) {
            largest = right;
        }

        if (largest != index) {
            std::swap(nodes[index], nodes[largest]);
            heapify_down(largest);
        }
    }; // End of Heap structure
};
// Main function to demonstrate how to use the heap
int main() {
    Heap heap;

    // Insert elements into the heap
    heap.insert(7);
    heap.insert(10);
    heap.insert(5);
    heap.insert(3);
    heap.insert(9);
    heap.insert(2);

    // Print the heap
    heap.print();   // Print heap: [ 10 9 7 3 5 2 ]

    // Remove and print the top element
    printf("Pop top: %d\n", heap.pop_top());  // Print: 10
    heap.print();   // Print heap: [ 9 5 7 3 2 ]

    // Remove and print the top element
    printf("Pop top: %d\n", heap.pop_top());  // Print: 9
    heap.print();   // Print heap: [ 7 5 2 3 ]

    // Insert a new element again
    heap.insert(6);
    heap.print();   // Print heap: [ 7 6 2 3 5 ]

    return 0;
}
