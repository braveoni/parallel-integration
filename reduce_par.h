#ifndef BARRIER_REDUCE_PAR_H
#define BARRIER_REDUCE_PAR_H

#endif //BARRIER_REDUCE_PAR_H

#if (defined(_MSVC_LANG) && _MSVC_LANG < 201700) || (!defined(_MSVC_LANG) && (__cplusplus < 201700 || (defined(__GNUC__) && __GNUC__ < 12)))
namespace std
{
    constexpr unsigned hardware_constructive_interference_size = 64u;
    constexpr unsigned hardware_destructive_interference_size = 64u;
}
#endif

template <class ElementType, class binary_fn>
ElementType reduce(ElementType* V, unsigned count, binary_fn f, ElementType zero)
{
    unsigned j = 1;
    while (count > j)
    {
        for (unsigned i = 0; i < count; i += j * 2)
        {
            ElementType other = zero;
            if (i + 1 < count)
                other = V[i + j];
            V[i] = f(V[i], other);
        }
        j *= 2;
    }
    return V[0];
}

#include <vector>
#include <thread>
//#include <barrier> //GCC 10
#include "barrier.h"
template <class ElementType, class binary_fn>
ElementType reduce_par(ElementType* V, unsigned count, binary_fn f, ElementType zero)
{
    constexpr unsigned k = 2;
    std::vector<std::thread> threads;
    unsigned T = get_num_threads();
    barrier bar{T};
    for (unsigned t = 1; t < T; t++)
        threads.emplace_back(std::thread{});
    auto thread_fn = [k, count, T, zero, V, f, &bar](unsigned t)
    {
        unsigned j = 1;
        while (count > j)
        {
            for (unsigned i = t * j * k; i < count; i += T * j * k)
            {
                ElementType other = zero;
                if (i + j < count)
                    other = V[i + j];
                V[i] = f(V[i], other);
            }
            j *= k;
            bar.arrive_and_wait();
        }
    };
    for (unsigned t = 1; t < T; t++)
        threads[t-1] = std::thread(thread_fn, t);
    thread_fn(0);
    for (auto& thread:threads)
        thread.join();
    return V[0];
}

unsigned get_num_threads();

//#include <concepts>
#include <type_traits>
template <class binary_fn, class unary_fn, class ElementType>
/*requires (
	std::is_invocable_r_v<ElementType,  binary_fn, ElementType, ElementType> &&
	std::is_invocable_r_v<ElementType, unary_fn, ElementType>
) */
auto reduce_par_2(binary_fn f, unary_fn get, ElementType x0, ElementType xn, ElementType step, ElementType zero)
{
    struct element_t
    {
        alignas(std::hardware_destructive_interference_size) ElementType value;
    };
    unsigned T = get_num_threads();
    static std::vector<element_t> reduction_buffer(std::thread::hardware_concurrency(), element_t{0.0});
    std::vector<std::thread> threads;
    barrier bar{T};
    auto thread_proc = [f, get, x0, xn, step, zero, T, &bar] (unsigned t)
    {
        std::size_t count = ElementType((xn-x0) / step);
        std::size_t nt = count / T, it0 = nt * t;
        ElementType my_result = zero;
        if (nt < (count % T))
            ++nt;
        else
            it0 += count % T;
        std::size_t it1 = it0 + nt;
        ElementType x = x0 + step * it0;
        for (std::size_t i = it0; i < it1; ++i, x += step)
            my_result = f(my_result, get(x));
        reduction_buffer[t].value = my_result;
        bar.arrive_and_wait();
        for (std::size_t reduction_distance = 1u, reduction_next = 2; reduction_distance < T; reduction_distance = reduction_next, reduction_next += reduction_next)
        {
            if (t + reduction_distance < T && (t & reduction_next - 1) == 0)
                reduction_buffer[t].value = f(reduction_buffer[t].value, reduction_buffer[t + reduction_distance].value);
            bar.arrive_and_wait();
        }
    };
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread:threads)
        thread.join();
    return reduction_buffer[0].value;
}