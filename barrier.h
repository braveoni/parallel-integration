#ifndef BARRIER_BARRIER_H
#define BARRIER_BARRIER_H

#endif //BARRIER_BARRIER_H

#include <mutex>
#include <condition_variable>

class latch
{
    unsigned m_T;
    std::condition_variable cv;
    std::mutex mtx;
public:
    latch(unsigned T):m_T(T) {}
    void arrive_and_wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        if (--m_T > 0)
        {
            while (m_T > 0)
                cv.wait(lock);
        }else
        {
            cv.notify_all();
        }
    }
};

class barrier
{
    const unsigned m_T_max;
    unsigned m_T;
    bool barrier_id = false;
    std::condition_variable cv;
    std::mutex mtx;
public:
    barrier(unsigned T):m_T_max(T), m_T(T) {}
    void arrive_and_wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        bool my_barrier_id = barrier_id;
        if (--m_T > 0)
        {
            while (my_barrier_id == barrier_id)
                cv.wait(lock);
        }else
        {
            cv.notify_all();
            m_T = m_T_max;
            barrier_id = !my_barrier_id;
        }
    }
};