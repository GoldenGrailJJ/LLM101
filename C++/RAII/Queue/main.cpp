#include <cstdio>
#include <memory>
#include <deque>

struct CircularQueue {
	std::unique_ptr<int[]> datas;
	size_t capacity;
	size_t front = 0;
	size_t rear = 0;
	size_t size = 0;

	CircularQueue(size_t cap) : capacity(cap) {
		datas = std::make_unique<int[]>(capacity);
	}

	void enqueue(int value) {
		if (size == capacity) {
			printf("Error: Queue is full!\n");
			return;
		}

		datas[rear] = value;
		rear = (rear + 1) % capacity;
		++size;
	}

	int dequeue() {
		if (size == 0) {
			printf("Error: Queue is empty!\n");
			return -1;
		}
		int value = datas[front];
		front = (front + 1) % capacity;
		--size;
		return value;
	}

	void print() const {
		printf("[ ");
		for (size_t i = 0; i < size; ++i) {
			printf("%d ", datas[(front + i) % capacity]);
		}
		printf("]\n");
	}

};

int main() {
	// -- 循环队列 --
	printf("CircularQueue:\n");
	CircularQueue queue(5);
	queue.enqueue(1);
	queue.enqueue(2);
	queue.enqueue(3);
	queue.enqueue(4);
	queue.enqueue(5);
	queue.print();  // 输出: [ 1 2 3 4 5 ]
	queue.enqueue(6);  // 错误，队列已满
	printf("Dequeue: %d\n", queue.dequeue());  // 输出: 1
	queue.print();  // 输出: [ 2 3 4 5 ]
	queue.enqueue(6);
	queue.print();  // 输出: [ 2 3 4 5 6 ]
}