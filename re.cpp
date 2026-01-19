rocblas_status rocblas_gemm_ex(
    rocblas_handle handle,       // rocblas 库的上下文句柄
    rocblas_operation transA,    // 
    rocblas_operation transB,    // 
    rocblas_int m,
    rocblas_int n,
    rocblas_int k,
    const void *alpha,
    const void *a,
    rocblas_datatype a_type,
    rocblas_int lda,
    const void *b,
    rocblas_datatype b_type,
    rocblas_int ldb,
    const void *c,
    rocblas_datatype c_type,
    rocblas_int ldc,
    const void *d,
    rocblas_datatype d_type,
    rocblas_int ldd,
    rocblas_datatype compute_type,
    rocblas_gemm_algo algo,
    int32_t solution_index,
    uint32_t flags)