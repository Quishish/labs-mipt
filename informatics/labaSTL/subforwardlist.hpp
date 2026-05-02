// subforward_list.hpp
#pragma once
#include <cstddef>

template<typename T>
class subforward_list {
private:
    struct Node {
        T data;
        Node* next;
        Node(const T& val) : data(val), next(nullptr) {}
    };

    Node* head;
    size_t sz;

public:
    // 🔹 Типы для совместимости со STL
    using value_type = T;
    using size_type = std::size_t;
    using iterator = Node*;
    using const_iterator = const Node*;

    // 🔹 Конструктор / деструктор
    subforward_list() : head(nullptr), sz(0) {}
    
    ~subforward_list() { clear(); }

    // 🔹 Базовые методы
    size_type size() const { return sz; }
    bool empty() const { return sz == 0; }

    iterator begin() { return head; }
    iterator end() { return nullptr; }
    const_iterator begin() const { return head; }
    const_iterator end() const { return nullptr; }

    // 🔹 push_front — основная операция для лабы
    void push_front(const T& val) {
        Node* new_node = new Node(val);
        new_node->next = head;
        head = new_node;
        ++sz;
    }

    // 🔹 pop_front — нужен для очистки в бенчмарке
    void pop_front() {
        if (!head) return;
        Node* tmp = head;
        head = head->next;
        delete tmp;
        --sz;
    }

    // 🔹 Очистка всего списка
    void clear() {
        while (head) {
            Node* tmp = head;
            head = head->next;
            delete tmp;
        }
        sz = 0;
    }
};