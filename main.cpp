#include <cstdlib>
#include <cstdio>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <iostream>

[[maybe_unused]] constexpr std::size_t CACHE_LINE = 64;

typedef float (*f_t) (float);

#define n 10000000u

void set_num_threads(unsigned T);

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

float integrate_omp(float a, float b, f_t f)
{
    double dx = (b - a) / n;
    std::unique_ptr<element_t[]> results;
    double res = 0.0;
    unsigned T;
#pragma omp parallel shared(results, T)
    {
        auto t = (unsigned)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned)get_num_threads();
            results = std::make_unique<element_t[]>(T);
        } //Барьер
        results[t].value = 0.0;
        for (size_t i = t; i < n; i += T)
            results[t].value += f((float)(dx * i + a));
    }
    for (size_t i = 0; i < T; ++i)
        res += results[i].value;
    return (float)(res * dx);
}

float integrate_cpp(float a, float b, f_t f)
{
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

float integrate_cpp_cs(float a, float b, f_t f)
{
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

class Iterator
{
    f_t f;
    double dx, a;
    unsigned i = 0;
public:
    typedef double value_type, * pointer, & reference;
    using iterator_category = std::input_iterator_tag;
    //Iterator() = default;
    Iterator(f_t fun, double delta_x, double x0, unsigned index) :f(fun), dx(delta_x), a(x0), i(index) {}
    [[nodiscard]] double value() const {
        return f(a + i * dx);
    }
    auto operator*() const { return this->value(); }
    Iterator& operator++()
    {
        ++i;
        return *this;
    }
    Iterator operator++(int)
    {
        auto old = *this;
        ++* this;
        return old;
    }
    bool operator==(const Iterator& other) const
    {
        return i == other.i;
    }
};

#include <numeric>

float integrate_cpp_reduce_1(float a, float b, f_t f)
{
#ifdef __GNUC__
    return 0.0;
#else
    double dx = (b - a) / n;
    return std::reduce(Iterator(f, dx, a, 0), Iterator(f, dx, a, n)) * dx;
#endif
}

#include "reduce_par.h"
float integrate_cpp_reduce_2(float a, float b, f_t f)
{
    double dx = (b - a) / n;
    return reduce_par_2([f, dx](double x, double y) {return x + y; }, f, (double)a, (double)b, (double)dx, 0.0) * dx;
}

float g(float x)
{
    return x * x;
}

typedef struct experiment_result_t_
{
    float result;
    double time;
} experiment_result_t;

typedef float (*integrate_t)(float a, float b, f_t f);
experiment_result_t run_experiment(integrate_t integrate)
{
    experiment_result_t result;
    double t0 = omp_get_wtime();
    result.result = integrate(-1, 1, g);
    result.time = omp_get_wtime() - t0;
    return result;
}

void run_experiments(experiment_result_t* results, float (*I) (float, float, f_t))
{
    for (unsigned T = 1; T <= std::thread::hardware_concurrency(); ++T)
    {
        set_num_threads(T);
        results[T - 1] = run_experiment(I);
    }
}

#include <iomanip>
void show_results_for(const char* name, const experiment_result_t* results)
{
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

int main(int argc, char** argv)
{

    //freopen("output.txt", "w", stdout);

    auto* results = (experiment_result_t*)malloc(get_num_threads() * sizeof(experiment_result_t));
    run_experiments(results, integrate_omp);
    show_results_for("integrate_omp", results);
    run_experiments(results, integrate_cpp);
    show_results_for("integrate_cpp", results);
    run_experiments(results, integrate_cpp_cs);
    show_results_for("integrate_cpp_cs", results);
    run_experiments(results, integrate_cpp_atomic);
    show_results_for("integrate_cpp_atomic", results);
    run_experiments(results, integrate_cpp_reduce_1);
    show_results_for("integrate_cpp_reduce_1", results);
    run_experiments(results, integrate_cpp_reduce_2);
    show_results_for("integrate_cpp_reduce_2", results);
    free(results);
    experiment_result_t r;
    r = run_experiment(integrate_omp);
    printf("integrate_omp. Result: %g. Time: %g\n", (double) r.result, r.time);
    r = run_experiment(integrate_cpp);
    printf("integrate_cpp. Result: %g. Time: %g\n", (double) r.result, r.time);
    r = run_experiment(integrate_cpp_cs);
    printf("integrate_cpp_cs. Result: %g. Time: %g\n", (double) r.result, r.time);
    r = run_experiment(integrate_cpp_atomic);
    printf("integrate_cpp_atomic. Result: %g. Time: %g\n", (double) r.result, r.time);
    r = run_experiment(integrate_cpp_reduce_2);
    printf("integrate_cpp_reduce_2. Result: %g. Time: %g\n", (double) r.result, r.time);

    return 0;
}