__device__ inline unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
    // 参数：address地址;compare:地址的旧值;val:地址的旧值+增量=新值
    // 这个地址不是临界区，只是使用原子操作保证写回操作的有效性
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    /*调用__atomic_compare_exchange_n函数如下所示*/
    /*
    ATOMIC();//原子操作，不可中断，值更改，要在临界区中进行
    int old_val = *address;
    if (old_val == compare) //在原子操作中获取到的地址的旧值，与调用atomicCAS之前获
    取到的旧值是否相同
    *address = val; //如果相同说明当前没有其他线程修改该地址的值，就将本线程
    的新值写回该地址；否则不写回
    END_ATOMIC();
    return old_val; //总是返回该地址在临界区时的旧值
    */
    return compare;
}
