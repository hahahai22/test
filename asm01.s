func(int, int, int):
        push    rbp							; 保存main的rbp（保存调用者的基地址）
        mov     rbp, rsp					; 为func函数设置新基指针
        mov     dword ptr [rbp - 4], edi
        mov     dword ptr [rbp - 8], esi
        mov     dword ptr [rbp - 12], edx
        mov     dword ptr [rbp - 16], 10
        mov     eax, dword ptr [rbp - 4]                                ; 返回值eax=x
        pop     rbp							; 恢复main的rbp
        ret									; 返回

main:
        push    rbp      					; 函数序言（prologue）。保存main的调用者基指针（便于返回时恢复）
        mov     rbp, rsp 					; 函数序言（prologue）。设置main函数基指针值为rsp的值
        sub     rsp, 16						; 为局部变量分配16字节栈空间
        mov     dword ptr [rbp - 4], 1		; 局部变量x：地址 rbp - 4
        mov     dword ptr [rbp - 8], 2		; 局部变量y
        mov     dword ptr [rbp - 12], 3		; 局部变量z 。分配了16字节，用了12字节，可能是为了栈对齐（16字节对齐）
        mov     edi, 1						; func函数参数放入寄存器 edi
        mov     esi, 2						; func函数参数放入寄存器 esi
        mov     edx, 3						; func函数参数放入寄存器 edx
        call    func(int, int, int)			; 在call之后，a. 将下一条指令的地址（返回地址）压入栈；b. 跳转到func函数。(1. call指令压入返回地址，rsp指向返回地址（在栈顶）2. func函数开始执行)
        xor     eax, eax					; 返回值0
        add     rsp, 16						; 释放局部变量空间
        pop     rbp
        ret


/*
这种机制的作用：
1. 独立的栈帧管理
        每个函数都需要有自己的基地址来访问：
        1. 局部变量
        2. 传入参数
        3. 返回地址
2. 调用链回溯 
3. 寄存器复用
整个系统只有一个物理RBP寄存器，但可以通过保存/恢复机制实现复用：
	进入函数时：保存前一个函数的RBP值到栈上
	设置新值：为当前函数创建新基地址
	退出函数时：从栈上恢复前一个函数的RBP值
*/
