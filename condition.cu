#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <cublasLt.h>

// __global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int64_t n)
// {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;
//     if (index < n) {
//         deviceC[index] = deviceA[index] + deviceB[index];
//     }
// }


// void launch_add(float *dA, float *dB, float *dC, int64_t n){
//     // 使用固定的配置，确保CUDA Graph可以捕获
//     const int blockSize = 8;
//     const int numBlocks = 4;
//     addKernel<<<numBlocks, blockSize>>>(dA, dB, dC, n);
// }


__global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int64_t n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        if (deviceA[index] > 0.5) {
            deviceC[index] = deviceA[index] + deviceB[index];
        } else {
            deviceC[index] = deviceA[index] - deviceB[index];
        }
    }
}


void launch_add(const at::Tensor& dA, const at::Tensor& dB, const at::Tensor& dC, int64_t n){
    // 使用固定的配置，确保CUDA Graph可以捕获
    const int blockSize = 8;
    const int numBlocks = 4;
    // at::cuda::CUDAGuard device_guard{(char)dA.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    addKernel<<<numBlocks, blockSize, 0, stream>>>(dA.data_ptr<float>(), dB.data_ptr<float>(), dC.data_ptr<float>(), n);
    // addKernel<<<numBlocks, blockSize>>>(dA.data_ptr<float>(), dB.data_ptr<float>(), dC.data_ptr<float>(), n);
}


void matrixMultiply(const at::Tensor& dA, const at::Tensor& dB, const at::Tensor& dC){
    const auto dA_sizes = dA.sizes();
    const auto dB_sizes = dB.sizes();
    const auto dC_sizes = dC.sizes();

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    int m = dA_sizes[0], n = dB_sizes[1], k = dA_sizes[1], d = dB_sizes[0];

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_T,CUBLAS_OP_N , d, m, n, &alpha, dB.data_ptr<float>(), n, dA.data_ptr<float>(), k, &beta, dC.data_ptr<float>(), d);

    cublasDestroy(handle);

}

//  a_sm = a_sm * (k.shape[-1] ** -0.5);
__global__ void scaled_vector(float* a_sm, float scale, int num_elements){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        a_sm[idx] *= scale;
    }

}


void stridedBatchedMatrixMultiply(const at::Tensor& dA, const at::Tensor& dB, at::Tensor& dC){
    const auto dA_sizes = dA.sizes();
    const auto dB_sizes = dB.sizes();
    const auto dC_sizes = dC.sizes();

    int m = dA_sizes[1], n = dB_sizes[2], k = dA_sizes[2], d = dB_sizes[1];
    int batchCount = dA_sizes[0];
    long int  strideA = m * k, strideB = n * d, strideC = m * d;
    // const float alpha = 1.0f;
    const float alpha = pow(n, -0.5);
    const float beta  = 0.0f;
   
    // float scale = powf(static_cast<float>(n), -0.5f);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_T,CUBLAS_OP_N , d, m, n, &alpha, dB.data_ptr<float>(), n, strideB, 
                         dA.data_ptr<float>(), k, strideA, &beta, dC.data_ptr<float>(), d, strideC, batchCount);
            
    cublasDestroy(handle);

                   
    // int num_elements = dC.numel();
    // int threads_per_block = 256;
    // int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;       
    // scaled_vector<<<num_blocks, threads_per_block>>>(dC.data_ptr<float>(), scale, num_elements);
    // cudaDeviceSynchronize();  // 确保内核函数执行完成
    
    // dC = dC * scale;

}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}


// void LtstridedBatchedMatrixMultiply(const at::Tensor& dA, const at::Tensor& dB, at::Tensor& dC, const float alpha, const float beta){
//     const auto dA_sizes = dA.sizes();
//     const auto dB_sizes = dB.sizes();
//     const auto dC_sizes = dC.sizes();
    
//     cublasLtHandle_t ltHandle = NULL;
//     cublasLtMatmulDesc_t operationDesc = NULL;
//     cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
//     cublasOperation_t transa = CUBLAS_OP_T;
//     cublasOperation_t transb = CUBLAS_OP_N;


//     int batchCount = dA_sizes[0];
//     int m = dA_sizes[1], k = dA_sizes[2], n = dB_sizes[1];
//     long int  stridea = n * k, strideb = m * k, stridec = m * n;
//     int lda = k, ldb = k , ldc = n;
//     // const float alpha = pow(k, -0.5);
//     // // const float alpha = 1.0f;
//     // const float beta  = 0.0f;



//     checkCublasStatus(cublasLtCreate(&ltHandle));
//     // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
//     // set the transforms for A and B
//     checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
//     checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
//     checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

//     // create matrix descriptors, we need to configure batch size and counts in this case
//     checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F,  k,  n, lda));
//     checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//     checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

//     checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k , m , ldb));
//     checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//     checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

//     checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, ldc));
//     checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
//     checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));

//     // in this simplified example we take advantage of cublasLtMatmul shortcut notation with algo=NULL which will force
//     // matmul to get the basic heuristic result internally. Downsides of this approach are that there is no way to
//     // configure search preferences (e.g. disallow tensor operations or some reduction schemes) and no way to store the
//     // algo for later use
//     checkCublasStatus(cublasLtMatmul(ltHandle,
//                                      operationDesc,
//                                      &alpha,
//                                      dB.data_ptr(),
//                                      Adesc,
//                                      dA.data_ptr(),
//                                      Bdesc,
//                                      &beta,
//                                      dC.data_ptr(),
//                                      Cdesc,
//                                      dC.data_ptr(),
//                                      Cdesc,
//                                      NULL,
//                                      NULL,
//                                      0,
//                                      0));

//     // descriptors are no longer needed as all GPU work was already enqueued
//     if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
//     if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
//     if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
//     if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

// }


void LtstridedBatchedMatrixMultiply(const at::Tensor& dA, const at::Tensor& dB, at::Tensor& dC, const float alpha, const float beta){
    const auto dA_sizes = dA.sizes();
    const auto dB_sizes = dB.sizes();
    const auto dC_sizes = dC.sizes();
    
    cublasLtHandle_t ltHandle = NULL;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;


    int batchCount = dA_sizes[0];
    int m = dA_sizes[1], k = dA_sizes[2], n = dB_sizes[2];
    long int  stridea = n * k, strideb = m * k, stridec = m * n;
    int lda = n, ldb = k , ldc = n;
    // const float alpha = pow(k, -0.5);
    // // const float alpha = 1.0f;
    // const float beta  = 0.0f;



    checkCublasStatus(cublasLtCreate(&ltHandle));
    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we need to configure batch size and counts in this case
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F,  n,  k, lda));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k , m , ldb));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, ldc));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));

    // in this simplified example we take advantage of cublasLtMatmul shortcut notation with algo=NULL which will force
    // matmul to get the basic heuristic result internally. Downsides of this approach are that there is no way to
    // configure search preferences (e.g. disallow tensor operations or some reduction schemes) and no way to store the
    // algo for later use
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     &alpha,
                                     dB.data_ptr(),
                                     Adesc,
                                     dA.data_ptr(),
                                     Bdesc,
                                     &beta,
                                     dC.data_ptr(),
                                     Cdesc,
                                     dC.data_ptr(),
                                     Cdesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

}

