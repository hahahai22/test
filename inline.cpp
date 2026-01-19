inline float atomicAdd(float *address, float val) // 参数列表：（参数地址，增量）
{
    unsigned int *uaddr{reinterpret_cast<unsigned int *>(address)}; // 地址
    unsigned int r{__atomic_load_n(uaddr, __ATOMIC_RELAXED)};       // 保存地址的旧值
    unsigned int old;
    do
    {
        old = __atomic_load_n(uaddr, __ATOMIC_RELAXED); // 获取当前循环的地址旧值
        if (r != old)
        {
            r = old;
            continue;
        } // 从多次执行循环的视角看，可以在每次循
        // 环时都保证r为该地址的旧值
        r = atomicCAS(uaddr, r, __float_as_uint(val + __uint_as_float(r)));
        // r为该地址在临界区中的旧值
        // 对应CAS中的判断，if (old_val == compare)如果从原子操作中得到地址的旧值，和原
        // 子操作之前的旧值相同，说明进行原子操作前后，这时没有其他线程修改该地址的值，此时CAS已经更改
        // 了地址的值。如果不相同，意味着另一个线程已经成功执行atomicCAS，CAS没有更改地址的值
        if (r == old)
            break;
    } while (true); // 循环一直执行，即等待着值没有被其他线程更改，本线程才更改
    return __uint_as_float(r);
}
