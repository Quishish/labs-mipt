// subvector.hpp
#pragma once
#include <cstddef>
#include <stdexcept>

template<typename T>
class subvector {
public:
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = std::size_t;

    // Конструктор
    subvector() : mas(nullptr), sz_(0), cap_(0) {}
    
    // Деструктор
    ~subvector() { clear(); }

    // 🔹 Accessors
    size_type size() const { return sz_; }
    size_type capacity() const { return cap_; }  // метод
    bool empty() const { return sz_ == 0; }

    // 🔹 Итераторы
    iterator begin() { return mas; }
    iterator end() { return mas + sz_; }
    const_iterator begin() const { return mas; }
    const_iterator end() const { return mas + sz_; }

    // 🔹 Доступ по индексу
    T& operator[](size_type idx) { return mas[idx]; }
    const T& operator[](size_type idx) const { return mas[idx]; }

    // 🔹 push_back
    void push_back(const T& val) {
        if (sz_ == cap_) {
            resize(cap_ == 0 ? 1 : cap_ * 2);
        }
        mas[sz_++] = val;
    }

    // 🔹 insert в произвольное место
    iterator insert(const_iterator pos, const T& val) {
        size_type idx = pos - mas;
        if (idx > sz_) idx = sz_;

        if (sz_ == cap_) {
            resize(cap_ == 0 ? 1 : cap_ * 2);
            pos = mas + idx;  // итераторы инвалидируются после realloc
        }
        iterator new_pos = mas + idx;

        // Сдвиг вправо
        for (size_type i = sz_; i > idx; --i) {
            mas[i] = mas[i - 1];
        }
        *new_pos = val;
        ++sz_;
        return new_pos;
    }

    // 🔹 erase из произвольного места
    iterator erase(const_iterator pos) {
        if (sz_ == 0) return end();
        size_type idx = pos - mas;
        if (idx >= sz_) return end();

        for (size_type i = idx; i < sz_ - 1; ++i) {
            mas[i] = mas[i + 1];
        }
        --sz_;
        return mas + idx;
    }

    // 🔹 Очистка
    void clear() {
        delete[] mas;
        mas = nullptr;
        sz_ = 0;
        cap_ = 0;
    }

private:
    T* mas;           // указатель на данные
    size_type sz_;    // текущий размер (бывшее top)
    size_type cap_;   // выделенная ёмкость (бывшее capacity)

    // 🔹 Внутренний метод изменения ёмкости
    bool resize(size_type new_cap) {
        if (new_cap == 0) {
            delete[] mas;
            mas = nullptr;
            sz_ = cap_ = 0;
            return true;
        }
        
        T* new_mas = new T[new_cap];
        size_type copy_count = (sz_ < new_cap) ? sz_ : new_cap;
        
        for (size_type i = 0; i < copy_count; ++i) {
            new_mas[i] = mas[i];
        }
        
        delete[] mas;
        mas = new_mas;
        cap_ = new_cap;
        
        if (new_cap < sz_) {
            sz_ = new_cap;
        }
        return true;
    }
};