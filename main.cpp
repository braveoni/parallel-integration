#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <iostream>


constexpr std::size_t CACHE_LINE = 64;

typedef float (*f_t) (float);

#define n 10000000u

std::atomic<unsigned> thread_num {std::thread::hardware_concurrency()};

void set_num_threads(unsigned T)
{
    thread_num = T;
    omp_set_num_threads(T);
}
unsigned get_num_threads()
{
    return thread_num;
}

typedef struct element_t_
{
    alignas(CACHE_LINE) double value;
} element_t;

#include <memory>

float integrate_seq(float a, float b, f_t f)
{
    double res = 0.;
    double dx = (b - a) / n;
    for (size_t i = 0; i < n; ++i)
        res += f((float) (dx * i + a));
    return (float) (res * dx);
}

float integrate_omp_fs(float a, float b, f_t f) // сам стандарт при парал реализации
{
    double dx = (b - a) / n;
    double* results;
    double res = 0.0;
    unsigned T;
#pragma omp parallel shared(results, T)
    {
        // t - индеф поток
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single // выполняется одним потоком один раз
        {
            T = (unsigned) omp_get_num_threads();
            results = (double*) calloc(sizeof(double), T); // массив у которого есть размер по памяти
            // results = std::make_unique<element_t[]>(T);
            if (!results)
                abort();
        }
        for (size_t i = t; i < n; i += T)
            results[t] += f((float) (dx * i + a));
    }
    for (size_t i = 0; i < T; ++i)
        res += results[i];
    free(results);
    return (float) (res * dx);
}

#ifndef __cplusplus
#ifdef _MSC_VER
#define alignas(X) __declspec(align(X))
#else
#include <stdalign.h>
#endif
#endif
#ifdef _MSC_VER
#define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

#define CACHE_LINE 64u

float integrate_omp(float a, float b, f_t f) // как cs, но с выравниванием
{
    double dx = (b - a) / n;
    std::unique_ptr<element_t[]> results; // по окончании функции results уничтожится
    double res = 0.0;
    unsigned T;
#pragma omp parallel shared(results, T)
    {
        unsigned t = (unsigned)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned)get_num_threads();
            results = std::make_unique<element_t[]>(T); // заполняет элемент
        }
        results[t].value = 0.0;
        for (size_t i = t; i < n; i += T)
            results[t].value += f((float)(dx * i + a));
    }
    for (size_t i = 0; i < T; ++i)
        res += results[i].value;
    return (float)(res * dx);
}

float integrate_omp_cs(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
#pragma omp parallel shared(res)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        double value = 0.0;
        for (size_t i = t; i < n; i += T)
        {
            value += f((float) (dx * i + a));
        }
#pragma omp critical
        {
            res+= value;
        }
    }
    return (float) (res * dx);
}

float integrate_omp_reduce(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
    int i;
#pragma omp parallel for reduction(+: res) schedule(static)
    for (i = 0; i < n; ++i)
        res += f((float) (dx * i + a));
    return (float) (res * dx);
}

float integrate_omp_reduce_dynamic(float a, float b, f_t f) // какой поток свободен, тот и будет работать
{
    double res = 0.0;
    double dx = (b - a) / n;
    int i;
#pragma omp parallel for reduction(+: res) schedule(dynamic)
    for (i = 0; i < n; ++i)
        res += f((float) (dx * i + a));
    return (float) (res * dx);
}

float integrate_mtx(float a, float b, f_t f)
{
    double res = 0.0;
    double dx = (b - a) / n;
    omp_lock_t mtx;
    omp_init_lock(&mtx);
#pragma omp parallel shared(res)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        double val = 0.0;
        for(size_t i = t; i < n; i+=T) {
            val += f(a + i * dx);
        }
        omp_set_lock(&mtx);
        res += val;
        omp_unset_lock(&mtx);
    }
    return res * dx;
}

float integrate_omp_atomic(float a, float b, f_t f) // более выгоден для атомарных операций
{
    double res = 0.0;
    double dx = (b - a) / n;
#pragma omp parallel shared(res)
    {
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        double val = 0.0;
        for (size_t i = t; i < n; i += T)
        {
            val += f((float) (dx * i + a));
        }
#pragma omp atomic
        res += val;
    }
    return (float) (res * dx);
}

float integrate_cpp(float a, float b, f_t f){
    double dx = (b - a) / n;
    unsigned T = get_num_threads();
    std::vector results(T, element_t{ 0.0 });
    auto thread_proc = [=, &results](unsigned t) {
        results[t].value = 0.0;
        for (size_t i = t; i < n; i += T)
            results[t].value += f((float)(dx * i + a));
    };
    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();
    double res = 0.0;
    for (size_t i = 0; i < T; ++i)
        res += results[i].value;
    return (float)(res * dx);
}

float integrate_cpp_cs(float a, float b, f_t f){
    double res = 0.0;
    double dx = (b - a) / n;
    unsigned T = get_num_threads();
    std::mutex mtx;
    auto thread_proc = [=, &res, &mtx](unsigned t) {
        double l_res = 0.0;
        for (size_t i = t; i < n; i += T)
            l_res += f((float)(dx * i + a));
        {
            std::scoped_lock lock(mtx);
            res += l_res;
        }
    };
    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();
    return res * dx;
}

float integrate_cpp_atomic(float a, float b, f_t f) //C++20
{
    std::atomic<double> res = 0.0;
    double dx = (b - a) / n;
    unsigned T = get_num_threads();
    auto thread_proc = [=, &res](unsigned t) {
        double l_res = 0.0;
        for (size_t i = t; i < n; i += T)
            l_res += f((float)(dx * i + a));
        res += l_res;
    };
    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();
    return res * dx;
}

#include <iterator>

class Iterator
{
    f_t f;
    double dx, a;
    unsigned i = 0;
public:
    typedef double value_type, *pointer, &reference;
    using iterator_category = std::input_iterator_tag;
    //Iterator() = default;
    Iterator(f_t fun, double delta_x, double x0, unsigned index):f(fun), dx(delta_x), a(x0), i(index) {}
    double value() const{
        return f(a + i * dx);
    }
    auto operator*() const {return this->value();}
    Iterator& operator++()
    {
        ++i;
        return *this;
    }
    Iterator operator++(int)
    {
        auto old = *this;
        ++*this;
        return old;
    }
    bool operator==(const Iterator& other) const
    {
        return i == other.i;
    }
};

#include "reduce_par.h"
float integrate_cpp_reduce_2(float a, float b, f_t f) {
    double dx = (b - a) / n;
    return reduce_par_2([f, dx](double x, double y) {return x + y; }, f, (double)a, (double)b, (double)dx, 0.0) * dx;
}

float g(float x) {
    return x * x;
}

typedef struct experiment_result_t_ {
    float result;
    double time;
} experiment_result_t;

typedef float (*integrate_t)(float a, float b, f_t f);
experiment_result_t run_experiment(integrate_t integrate) {
    experiment_result_t result;
    double t0 = omp_get_wtime();
    result.result = integrate(-1, 1, g);
    result.time = omp_get_wtime() - t0;
    return result;
}

void run_experiments(experiment_result_t* results, float (*I) (float, float, f_t)) {
    for (unsigned T = 1; T <= std::thread::hardware_concurrency(); ++T) {
        set_num_threads(T);
        results[T - 1] = run_experiment(I);
    }
}

#include <iomanip>
void show_results_for(const char* name, const experiment_result_t* results) {
    unsigned w = 10;
    std::cout << name << "\n";
    std::cout << std::setw(w) << "T" << "\t"
              << std::setw(w) << "Time" << "\t"
              << std::setw(w) << "Result" << "\t"
              << std::setw(w) << "Speedup\n";
    for (unsigned T = 1; T <= omp_get_num_procs(); T++)
        std::cout << std::setw(w) << T << "\t"
                  << std::setw(w) << results[T - 1].time << "\t"
                  << std::setw(w) << results[T - 1].result << "\t"
                  << std::setw(w) << results[0].time / results[T - 1].time << "\n";
};

int main(int argc, char** argv) {

    experiment_result_t* results = (experiment_result_t*)malloc(get_num_threads() * sizeof(experiment_result_t));

    run_experiments(results, integrate_seq);
    show_results_for("integrate_seq", results);
    run_experiments(results, integrate_omp_fs);
    show_results_for("integrate_omp_fs", results);
    run_experiments(results, integrate_omp);
    show_results_for("integrate_omp", results);
    run_experiments(results, integrate_omp_cs);
    show_results_for("integrate_omp_cs", results);
    run_experiments(results, integrate_mtx);
    show_results_for("integrate_mtx", results);
    run_experiments(results, integrate_omp_atomic);
    show_results_for("integrate_omp_atomic", results);
    run_experiments(results, integrate_omp_reduce);
    show_results_for("integrate_omp_reduce", results);
    run_experiments(results, integrate_omp_reduce_dynamic);
    show_results_for("integrate_omp_reduce_dynamic", results);

    run_experiments(results, integrate_cpp);
    show_results_for("integrate_cpp", results);
    run_experiments(results, integrate_cpp_cs);
    show_results_for("integrate_cpp_cs", results);
    run_experiments(results, integrate_cpp_atomic);
    show_results_for("integrate_cpp_atomic", results);
    run_experiments(results, integrate_cpp_reduce_2);
    show_results_for("integrate_cpp_reduce_2", results);
    free(results);
    experiment_result_t r;

    return 0;
}