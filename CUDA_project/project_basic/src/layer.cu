/* Last Updated: 24.08.27. 18:30 */
#include "layer.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[m * N + n] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[m * N + n] += in->buf[m * K + k] * w->buf[n * K + k];
      }
      out->buf[m * N + n] += b->buf[n];
    }
  }
}

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

// static float *A_gpu, *B_gpu, *C_gpu, *bias_gpu;

__global__ void Linear_kernel(float *A, float *B, float *C, int M, int N, int K, float * Bias){
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int gj = blockIdx.x; int gi = blockIdx.y;
  if(gi * BLOCK_SIZE_X >= M || gj * BLOCK_SIZE_Y >= N) return;
  int lj = threadIdx.x; int li = threadIdx.y;

  __shared__ float Alocal[BLOCK_SIZE_Y][BLOCK_SIZE_X];
  __shared__ float Blocal[BLOCK_SIZE_X][BLOCK_SIZE_Y];

  float c = 0.f;

  int A_row_index = (gi*BLOCK_SIZE_X + li);
  int B_row_index = (gj*BLOCK_SIZE_Y + lj);

  for(int bk = 0; bk<K; bk += BLOCK_SIZE_X){
    int A_col_index = bk + lj;
    Alocal[li][lj] = (A_row_index < M && A_col_index < K)? A[A_row_index * K + A_col_index] : 0.f;

    int B_col_index = bk + li;
    Blocal[li][lj] = (B_row_index < N && B_col_index < K)? B[B_row_index * K + B_col_index] : 0.f;
    __syncthreads();

    for(int lk =0;lk<BLOCK_SIZE_Y; ++lk){
      c += Alocal[li][lk] * Blocal[lk][lj];
    }
    __syncthreads();
  }
  if (i < M && j < N) C[i * N + j] = c + Bias[j];
}

void Linear_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  dim3 blockDim(BLOCK_SIZE_Y, BLOCK_SIZE_X);
  dim3 gridDim((N+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y, (M+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
  Linear_kernel<<<gridDim, blockDim>>>(in->gbuf, w->gbuf, out->gbuf, M,N,K, b->gbuf);
  CHECK_CUDA(cudaGetLastError());

}
float *A_gpu, *B_gpu, *C_gpu, *bias_gpu;
void Linear_cuda_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  // init
  CHECK_CUDA(cudaMalloc(&A_gpu, M*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&bias_gpu, N*sizeof(float)));

  CHECK_CUDA(cudaDeviceSynchronize());

  // upload A, B to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, in->buf, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, w->buf,  N*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(bias_gpu, b->buf,  N*sizeof(float), cudaMemcpyHostToDevice));


  // kernel on a GPU
  dim3 blockDim(BLOCK_SIZE_Y, BLOCK_SIZE_X);
  dim3 gridDim((N+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y, (M+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);
  Linear_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M,N,K, bias_gpu);
  CHECK_CUDA(cudaGetLastError());


  // Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(out->buf, C_gpu, M*N*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());

  // free
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaDeviceSynchronize());

}

/* Reshape 
 * @param [in]   in: [N, D]
 * @param [out] out: [N, C, H, W]
 * 'N' is the number of input tensors.
 * 'D' is the dimension of the input tensor.
 * 'C' is the number of channels.
 * 'H' is the height of the output tensor.
 * 'W' is the width of the output tensor.
 */
void Reshape(Tensor *in, Tensor *out) {
  size_t N = in->shape[0];
  size_t D = in->shape[1];
  size_t C = out->shape[1];
  size_t H = out->shape[2];
  size_t W = out->shape[3];

   // printf("\nreshape");


  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          out->buf[n * C * H * W + c * H * W + h * W + w] =
              in->buf[n * D + c * H * W + h * W + w];
        }
      }
    }
  }
}

float *Reshape_I_gpu, *Reshape_O_gpu;

__global__ void Reshape_kernel(float *in_buf, float *out_buf, 
                               int N, int D, int C, int H, int W){
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;                               
  const int on = tidx / (C*H*W);
  const int oc = (tidx / (H*W)) % C;
  const int oh = (tidx / W) % H;
  const int ow = tidx % W;              

  if(on >= N || oc >= C || oh >= H || ow >= W) return;

  out_buf[on*C*H*W + oc*H*W + oh*W + ow] = 
    in_buf[on*D + oc*H*W + oh*W + ow];
}

void Reshape_cuda(Tensor *in, Tensor *out) {
  size_t N = in->shape[0];
  size_t D = in->shape[1];
  size_t C = out->shape[1];
  size_t H = out->shape[2];
  size_t W = out->shape[3];

  // 초기화
  // CHECK_CUDA(cudaMalloc(&Reshape_I_gpu, N * D * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&Reshape_O_gpu, N * C * H * W * sizeof(float)));

  // input : cpu->gpu
  // CHECK_CUDA(cudaMemcpy(Reshape_I_gpu, in->buf, N * D * sizeof(float), cudaMemcpyHostToDevice));

  // 전달
  int total_threads = N * C * H * W;
  int block_size = 64;

  dim3 blockDim(block_size);
  dim3 gridDim((total_threads + block_size - 1) / block_size);

  Reshape_kernel<<<gridDim, blockDim>>>(Reshape_I_gpu, Reshape_O_gpu, N, D, C, H, W);

  CHECK_CUDA(cudaMemcpy(out->buf, Reshape_O_gpu, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(Reshape_I_gpu));
  CHECK_CUDA(cudaFree(Reshape_O_gpu));
  CHECK_CUDA(cudaDeviceSynchronize());
}


/* ConvTranspose2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C, K, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *    
 *    OH = (H - 1) * stride - 2 * pad + dilation * (R - 1) + output_pad + 1
 *    OW = (W - 1) * stride - 2 * pad + dilation * (S - 1) + output_pad + 1
 *    In this model, R = S = 3, stride = 2, pad = 1, dilation = 1, output_pad = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 */
void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[1];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];
 
  const size_t stride = 2;
  const size_t pad = 1;
  const size_t dilation = 1;

  for (size_t oc = 0; oc < K; ++oc) {
    for (size_t oh = 0; oh < OH; ++oh) {
      for (size_t ow = 0; ow < OW; ++ow) {
        float o = bias->buf[oc];
        for (size_t c = 0; c < C; ++c) {
          for (size_t r = 0; r < R; ++r) {
            for (size_t s = 0; s < S; ++s) {
              if ((oh - (r * dilation - pad)) % stride != 0) continue;
              if ((ow - (s * dilation - pad)) % stride != 0) continue;
              size_t h = (oh - (r * dilation - pad)) / stride;
              size_t w = (ow - (s * dilation - pad)) / stride;
              if (h >= H || w >= W) continue;
              o += in->buf[c * H * W + h * W + w] * 
                weight->buf[c * K * R * S + oc * R * S + r * S + s];
            }
          }
        }
        out->buf[oc * OH * OW + oh * OW + ow] = o;
      }
    }
  }
}

__global__ void ConvTranspose2d_kernel(float *i_gpu, float *o_gpu, float *f_gpu, 
                              int N, int C, int H, int W, int K, int R, int S, int OH, int OW,
                              int stride, int pad, int dilation, float *bias_gpu){

  const int ON = N;
  const int OC = K;
  
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int on = tidx / (OC * OH * OW);
  const int oc = (tidx / (OH * OW)) % OC;
  const int oh = (tidx / OW) % OH;
  const int ow = tidx % OW;

  if (on >= ON || oc >= OC || oh >= OH || ow >= OW) return;

  float sum = bias_gpu[oc];
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        if ((oh - (r * dilation - pad)) % stride != 0) continue;
        if ((ow - (s * dilation - pad)) % stride != 0) continue;
        const int h = (oh - (r * dilation - pad)) / stride;
        const int w = (ow - (s * dilation - pad)) / stride;
        if (h >= H || w >= W) continue;
        sum += i_gpu[c * H * W + h * W + w] * f_gpu[c * K * R * S + oc * R * S + r * S + s];
      }
    }
  }
  o_gpu[oc * OH * OW + oh * OW + ow] = sum;

}

void ConvTranspose2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[1];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];

  const size_t stride = 2;
  const size_t pad = 1;
  const size_t dilation = 1;
    //printf("\nconvtrans2d");


  // 전달
  int total_threads = N*K*OH*OW;
  int block_size = 64;

  dim3 blockDim(block_size);
  dim3 gridDim((total_threads + block_size - 1) / block_size);

  ConvTranspose2d_kernel<<<gridDim, blockDim>>>(in->gbuf, out->gbuf, weight->gbuf, 
                                      N, C, H, W, K, R, S, OH, OW,
                                      stride, pad, dilation, bias->gbuf);
  
  CHECK_CUDA(cudaDeviceSynchronize());
}


/* BatchNorm2d (track_running_stats=False)
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C]
 * @param [in3]   bias: [C]
 * @param [out]    out: [N, C, H, W]  
 * 
 *    out = weight * (in - mean) / sqrt(var + 1e-5) + bias 
 * 
 * 'N' is the number of input tensors.
 * 'C' is the number of channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 */
void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];

  const float eps = 1e-5f;

  for (size_t c = 0; c < C; c++) {
    // 1. Caculate mean for each channel
    float mean = 0.0f;
    float var = 0.0f;
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        float val = in->buf[c * H * W + h * W + w];
        mean += val;
      }
    }
    mean /= (H * W);

    // 2. Caculate variance for each channel
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        float val = in->buf[c * H * W + h * W + w];
        var += (val - mean) * (val - mean);
      }
    }
    var /= (H * W);

    // 3. Normalize with the calculated mean and variance
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        out->buf[c * H * W + h * W + w] =
          weight->buf[c] * 
          (in->buf[c * H * W + h * W + w] - mean) /
          sqrt(var + eps) +
          bias->buf[c];
      }
    }
  }
}

__global__ void BatchNorm2d_kernel(const float *in, const float *weight, const float *bias, float *out, int C, int H, int W) {
    const float eps = 1e-5f;

    int c = blockIdx.x;  // 채널 인덱스
    int h = threadIdx.y; // 높이 인덱스
    int w = threadIdx.x; // 너비 인덱스

    // 인덱스 계산
    int index = c * H * W + h * W + w;

    // 1. Calculate mean for the channel
    float mean = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            mean += in[c * H * W + i * W + j];
        }
    }
    mean /= (H * W);

    // 2. Calculate variance for the channel
    float var = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float val = in[c * H * W + i * W + j];
            var += (val - mean) * (val - mean);
        }
    }
    var /= (H * W);

    // 3. Normalize with the calculated mean and variance
    out[index] = weight[c] * (in[index] - mean) / sqrtf(var + eps) + bias[c];
}

void BatchNorm2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];

    dim3 blockDim(W, H);  // 각 스레드는 하나의 픽셀을 담당
    dim3 gridDim(C);      // 각 블록은 하나의 채널을 담당

    BatchNorm2d_kernel<<<gridDim, blockDim>>>(in->gbuf, weight->gbuf, bias->gbuf, out->gbuf, C, H, W);
    CHECK_CUDA(cudaDeviceSynchronize());
}
/* LeakyReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  const float alpha = 0.01;

  for (size_t i = 0; i < N; i++) {
    if (inout->buf[i] < 0) { inout->buf[i] *= alpha; }
  }
}

/* LeakyReLU GPU kernel
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void LeakyReLU_kernel(float *inout, size_t N, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { 
    if (inout[idx] < 0) { inout[idx] *= alpha; }
  }
}  

/* LeakyReLU using CUDA GPU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU_cuda(Tensor *inout) {
  size_t N = inout->num_elem();

  const float alpha = 0.01;
  LeakyReLU_kernel<<<(N + 255) / 256, 256>>>(inout->gbuf, N, alpha);
  CHECK_CUDA(cudaDeviceSynchronize());

}

/* Conv2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [K, C, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *
 *   OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1
 *   OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 *   In this model, R = S = 3, stride = 1, pad = 1, dilation = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 */
void Conv2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[0];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];

  const size_t stride = 1;
  const size_t pad = 1;
  const size_t dilation = 1;

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < K; oc++) {
      for (size_t oh = 0; oh < OH; oh++) {
        for (size_t ow = 0; ow < OW; ow++) {
          float o = bias->buf[oc];
          for (size_t c = 0; c < C; c++) {
            for (size_t r = 0; r < R; r++) {
              for (size_t s = 0; s < S; s++) {
                size_t h = oh * stride - pad + r * dilation;
                size_t w = ow * stride - pad + s * dilation;
                if (h >= H || w >= W) continue;
                o += in->buf[n * C * H * W + c * H * W + h * W + w] *
                  weight->buf[oc * C * R * S + c * R * S + r * S + s];
              }
            }
          }
          out->buf[n * K * OH * OW + oc * OH * OW + oh * OW + ow] = o;
        }
      }
    }
  }
}

// float *I_gpu, *F_gpu, *O_gpu, *Bias_gpu;

__global__ void Conv2d_kernel(float *I_gpu_1, float *O_gpu_1, float *F_gpu_1, 
                              int N, int C, int H, int W, int K, int R, int S, int OH, int OW,
                              int stride, int pad, int dilation, float *Bias_gpu_1){

  const int ON = N;
  const int OC = K;
  OH = 1 + (H + 2 * pad - (((R - 1) * dilation) + 1)) / stride;
  OW = 1 + (W + 2 * pad - (((S - 1) * dilation) + 1)) / stride;
  
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int on = tidx / (OC * OH * OW);
  const int oc = (tidx / (OH * OW)) % OC;
  const int oh = (tidx / OW) % OH;
  const int ow = tidx % OW;

  if (on >= ON || oc >= OC || oh >= OH || ow >= OW) return;


  float sum = Bias_gpu_1[oc];
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        const int n = on;
        const int h = oh * stride - pad + r * dilation;
        const int w = ow * stride - pad + s * dilation;
        const int k = oc;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        sum += I_gpu_1[((n * C + c) * H + h) * W + w] * F_gpu_1[((k * C + c) * R + r) * S + s];
      }
    }
  }
  //O_gpu_1[((on * OC + oc) * OH + oh) * OW + ow] = sum;
}

void Conv2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[0];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];

  const size_t stride = 1;
  const size_t pad = 1;
  const size_t dilation = 1;
  // 전달
  int total_threads = N * C * H * W;
  int block_size = 64;

  dim3 blockDim(block_size);
  dim3 gridDim((total_threads + block_size - 1) / block_size);

  Conv2d_kernel<<<gridDim, blockDim>>>(in->gbuf, out->gbuf, weight->gbuf, 
                                      N, C, H, W, K, R, S, OH, OW,
                                      stride, pad, dilation, bias->gbuf);

  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Tanh 
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void Tanh(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = tanh(inout->buf[i]);
  }
}

__global__ void Tanh_kernel(float * inout, int N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if(tidx < N)
    inout[tidx] = tanh(inout[tidx]);
}

void Tanh_cuda(Tensor *inout)
{
  // float * inout_float;
  int N = inout->num_elem();
  dim3 blockDim(32);
  dim3 gridDim((N + 32 - 1) / 32);
  Tanh_kernel<<<gridDim, blockDim>>>(inout->gbuf, N);
  cudaDeviceSynchronize();
}
