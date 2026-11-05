#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>

// Exact finite E4M3FN decoding. Scales are applied in F32 after each K=128 block.
__device__ float e4m3(unsigned char b) {
  int e=(b>>3)&15, m=b&7;
  float v=e==0 ? ldexpf(float(m),-9) : (e==15 && m==7 ? NAN : ldexpf(1.f+float(m)*.125f,e-7));
  return (b&128) ? -v:v;
}

// Decode/verification: each warp cooperates on K for one output channel and
// reuses each loaded weight across up to four M rows. No expanded weight buffer.
template<class T> __device__ void mv(const T* x,const unsigned char* w,const float* s,T* y,int M,int N,int K) {
  int lane=threadIdx.x%32, n=blockIdx.x*8+threadIdx.x/32;
  int m0=blockIdx.y*4;
  float acc[4]={0,0,0,0};
  for(int kb=0;kb<K;kb+=128) {
    float part[4]={0,0,0,0};
    for(int k=kb+lane;k<kb+128 && k<K;k+=32) {
      float v=n<N?e4m3(w[(size_t)n*K+k]):0.f;
      #pragma unroll
      for(int m=0;m<4;++m) if(m0+m<M) part[m]=fmaf(float(x[(size_t)(m0+m)*K+k]),v,part[m]);
    }
    float scale=n<N?s[(n/128)*((K+127)/128)+kb/128]:0.f;
    #pragma unroll
    for(int m=0;m<4;++m) acc[m]=fmaf(part[m],scale,acc[m]);
  }
  #pragma unroll
  for(int m=0;m<4;++m) {
    for(int d=16;d;d>>=1) acc[m]+=__shfl_down_sync(0xffffffff,acc[m],d);
    if(lane==0 && n<N && m0+m<M) y[(size_t)(m0+m)*N+n]=T(acc[m]);
  }
}

// SM80 floor: E4M3FN is decoded in shared memory; the MMA instructions operate
// on F16/BF16, NOT native FP8. A 16x64 output tile shares A across four warps and
// shares each B tile across 16 M rows. FP32 partial sums use direct source scales.
template<class T> __device__ void mm(const T* x,const unsigned char* w,const float* s,T* y,int M,int N,int K) {
  using namespace nvcuda;
  __shared__ __align__(32) T a[16*16];
  __shared__ __align__(32) T b[64*16];
  __shared__ __align__(32) float out[4*16*16];
  int warp=threadIdx.x/32, m0=blockIdx.y*16,n0=blockIdx.x*64;
  wmma::fragment<wmma::accumulator,16,16,16,float> total, part;
  wmma::fill_fragment(total,0.f);
  for(int kb=0;kb<K;kb+=128) {
    wmma::fill_fragment(part,0.f);
    for(int ki=0;ki<128;ki+=16) {
      for(int i=threadIdx.x;i<256;i+=128) {
        int m=m0+i/16,k=kb+ki+i%16;
        a[i]=(m<M && k<K)?x[(size_t)m*K+k]:T(0.f);
      }
      for(int i=threadIdx.x;i<1024;i+=128) {
        int n=n0+i/16,k=kb+ki+i%16;
        b[i]=T((n<N && k<K)?e4m3(w[(size_t)n*K+k]):0.f);
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> bf;
      wmma::load_matrix_sync(af,a,16);
      wmma::load_matrix_sync(bf,b+warp*256,16);
      wmma::mma_sync(part,af,bf,part);
      __syncthreads();
    }
    float scale=s[(n0/128)*((K+127)/128)+kb/128];
    #pragma unroll
    for(int i=0;i<total.num_elements;++i) total.x[i]=fmaf(part.x[i],scale,total.x[i]);
  }
  wmma::store_matrix_sync(out+warp*256,total,16,wmma::mem_row_major);
  __syncthreads();
  for(int i=threadIdx.x;i<1024;i+=128) {
    int warp_i=i/256,j=i%256,m=m0+j/16,n=n0+warp_i*16+j%16;
    if(m<M && n<N) y[(size_t)m*N+n]=T(out[i]);
  }
}
// Vectorized decode GEMV (M <= MR, MR = 4 or 8): each lane loads 16 E4M3 bytes at a time and
// decodes them exactly through F16 lanes (byte moved to the high byte, exponent
// and mantissa shifted down one bit = value * 2^-8; the x256 is applied once per
// output). Activations are read as 16-byte vectors straight from global memory
// (L1/L2 resident across the block's warps). One warp per output channel, eight
// channels per block, F32 accumulation with the block scale applied once per
// 16-byte chunk. Requires K % 128 == 0 and 16-byte aligned rows.
__device__ __forceinline__ float fp8_dot4(unsigned w, const float* x) {
  unsigned lo = __byte_perm(w, 0u, 0x1404);
  unsigned hi = __byte_perm(w, 0u, 0x3424);
  lo = (lo & 0x80008000u) | ((lo & 0x7F007F00u) >> 1);
  hi = (hi & 0x80008000u) | ((hi & 0x7F007F00u) >> 1);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
  float acc = a.x * x[0];
  acc = fmaf(a.y, x[1], acc);
  acc = fmaf(b.x, x[2], acc);
  return fmaf(b.y, x[3], acc);
}
// fp8_dot4 in two halves: decode four bytes once (exactly as fp8_dot4 does),
// then the same multiply-add chain against any number of activation rows.
__device__ __forceinline__ void fp8_decode4(unsigned w, float* out) {
  unsigned lo = __byte_perm(w, 0u, 0x1404);
  unsigned hi = __byte_perm(w, 0u, 0x3424);
  lo = (lo & 0x80008000u) | ((lo & 0x7F007F00u) >> 1);
  hi = (hi & 0x80008000u) | ((hi & 0x7F007F00u) >> 1);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
  out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
}
__device__ __forceinline__ float dot4(const float* w, const float* x) {
  float acc = w[0] * x[0];
  acc = fmaf(w[1], x[1], acc);
  acc = fmaf(w[2], x[2], acc);
  return fmaf(w[3], x[3], acc);
}
template<class T> __device__ __forceinline__ void load16(const T* x, float* out) {
  const uint4 a = *reinterpret_cast<const uint4*>(x);
  const uint4 b = *reinterpret_cast<const uint4*>(x + 8);
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  #pragma unroll
  for(int i=0;i<8;++i) { out[i]=float(pa[i]); out[8+i]=float(pb[i]); }
}
// Each weight chunk is loaded once for all MR rows, and every row sums in the
// same order (lane sweeps of 512, then a butterfly) whatever M is, so a row's
// bits do not depend on its batch: the MR = 8 form serves 5 to 8 rows with the
// same bandwidth-bound shape instead of the tile kernel below.
template<class T,int MR> __device__ void mv2(const T* x,const unsigned char* w,const float* s,T* y,int M,int N,int K) {
  const int lane=threadIdx.x&31, n=blockIdx.x*8+(threadIdx.x>>5);
  if(n>=N) return;
  const unsigned char* row=w+(size_t)n*K;
  const float* scales=s+(size_t)(n>>7)*(K>>7);
  float acc[MR];
  #pragma unroll
  for(int m=0;m<MR;++m) acc[m]=0.f;
  #pragma unroll 4
  for(int k0=lane*16;k0<K;k0+=512) {
    const uint4 q=*reinterpret_cast<const uint4*>(row+k0);
    const float scale=scales[k0>>7];
    // Decode the chunk once for every row: the same values and the same
    // multiply-add order as fp8_dot4 per row, so the bits are unchanged.
    float wv[16];
    fp8_decode4(q.x,wv);
    fp8_decode4(q.y,wv+4);
    fp8_decode4(q.z,wv+8);
    fp8_decode4(q.w,wv+12);
    #pragma unroll
    for(int m=0;m<MR;++m) {
      if(m<M) {
        float xv[16];
        load16(x+(size_t)m*K+k0,xv);
        float part=dot4(wv,xv);
        part+=dot4(wv+4,xv+4);
        part+=dot4(wv+8,xv+8);
        part+=dot4(wv+12,xv+12);
        acc[m]=fmaf(part,scale,acc[m]);
      }
    }
  }
  #pragma unroll
  for(int m=0;m<MR;++m) {
    for(int d=16;d;d>>=1) acc[m]+=__shfl_down_sync(0xffffffff,acc[m],d);
    if(lane==0 && m<M) y[(size_t)m*N+n]=T(acc[m]*256.f);
  }
}
// Long-span prefill: output rows [row0, row0 + rows) of a block-FP8 matrix
// expanded to T (value * block scale, rounded once) into a [rows, K] tile that
// a dense GEMM (cuBLAS) then multiplies. Elementwise: each thread expands four
// consecutive bytes of one row, which share one block scale (K % 128 == 0).
template<class T> __device__ void dequant_rows(const unsigned char* w,const float* s,T* out,int row0,int rows,int K) {
  const size_t i=((size_t)blockIdx.x*blockDim.x+threadIdx.x)*4;
  if(i>=(size_t)rows*K) return;
  const int r=(int)(i/(size_t)K), k=(int)(i%(size_t)K), row=row0+r;
  const float scale=s[(size_t)(row>>7)*(K>>7)+(k>>7)];
  const unsigned char* src=w+(size_t)row*K+k;
  #pragma unroll
  for(int j=0;j<4;++j) out[i+j]=T(e4m3(src[j])*scale);
}
extern "C" __global__ void qwen38_fp8_dequant_rows_f16(const unsigned char*w,const float*s,__half*out,int row0,int rows,int K){dequant_rows(w,s,out,row0,rows,K);}
extern "C" __global__ void qwen38_fp8_dequant_rows_bf16(const unsigned char*w,const float*s,__nv_bfloat16*out,int row0,int rows,int K){dequant_rows(w,s,out,row0,rows,K);}
#define EXPORT(T,S) \
extern "C" __global__ void qwen38_fp8_mv_##S(const T*x,const unsigned char*w,const float*s,T*y,int M,int N,int K){mv(x,w,s,y,M,N,K);} \
extern "C" __global__ void qwen38_fp8_mm_##S(const T*x,const unsigned char*w,const float*s,T*y,int M,int N,int K){mm(x,w,s,y,M,N,K);}
EXPORT(__half,f16)
EXPORT(__nv_bfloat16,bf16)
extern "C" __global__ void qwen38_fp8_mv2_f16(const __half*x,const unsigned char*w,const float*s,__half*y,int M,int N,int K){mv2<__half,4>(x,w,s,y,M,N,K);}
extern "C" __global__ void qwen38_fp8_mv2_bf16(const __nv_bfloat16*x,const unsigned char*w,const float*s,__nv_bfloat16*y,int M,int N,int K){mv2<__nv_bfloat16,4>(x,w,s,y,M,N,K);}
extern "C" __global__ void qwen38_fp8_mv2x8_f16(const __half*x,const unsigned char*w,const float*s,__half*y,int M,int N,int K){mv2<__half,8>(x,w,s,y,M,N,K);}
extern "C" __global__ void qwen38_fp8_mv2x8_bf16(const __nv_bfloat16*x,const unsigned char*w,const float*s,__nv_bfloat16*y,int M,int N,int K){mv2<__nv_bfloat16,8>(x,w,s,y,M,N,K);}
extern "C" __global__ void qwen38_fp8_mv_f32(const float*x,const unsigned char*w,const float*s,float*y,int M,int N,int K){mv(x,w,s,y,M,N,K);}

// Vectorized decode GEMV over F32 activations and outputs, for decoders that
// keep an F32 residual stream: one warp per output channel, eight channels a
// block, up to four rows a block (grid.y covers more rows). Each lane loads
// 16 E4M3 bytes at a time and 16 activations as four float4; every row sums
// its dot product in the same order (lane sweeps of 512, then a butterfly)
// whatever the row count, so a row's bits never depend on its batch.
// epilogue 1 applies relu(y)^2 (a relu^2 MLP's up projection). Requires
// K % 128 == 0 and 16-byte aligned rows.
extern "C" __global__ void izwi_fp8_mv_f32(const float* __restrict__ x,
                                           const unsigned char* __restrict__ w,
                                           const float* __restrict__ s,
                                           float* __restrict__ y, int M, int N, int K,
                                           int epilogue) {
  const int lane = threadIdx.x & 31, n = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int m0 = blockIdx.y * 4;
  if (n >= N) return;
  const unsigned char* row = w + (size_t)n * K;
  const float* scales = s + (size_t)(n >> 7) * (K >> 7);
  float acc[4] = {0, 0, 0, 0};
  for (int k0 = lane * 16; k0 < K; k0 += 512) {
    const uint4 q = *reinterpret_cast<const uint4*>(row + k0);
    const float scale = scales[k0 >> 7];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      if (m0 + m < M) {
        const float* xr = x + (size_t)(m0 + m) * K + k0;
        float xv[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float4 f = *reinterpret_cast<const float4*>(xr + 4 * i);
          xv[4 * i] = f.x;
          xv[4 * i + 1] = f.y;
          xv[4 * i + 2] = f.z;
          xv[4 * i + 3] = f.w;
        }
        float part = fp8_dot4(q.x, xv);
        part += fp8_dot4(q.y, xv + 4);
        part += fp8_dot4(q.z, xv + 8);
        part += fp8_dot4(q.w, xv + 12);
        acc[m] = fmaf(part, scale, acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    for (int d = 16; d; d >>= 1) acc[m] += __shfl_down_sync(0xffffffff, acc[m], d);
    if (lane == 0 && m0 + m < M) {
      float v = acc[m] * 256.f;
      if (epilogue == 1) {
        v = fmaxf(v, 0.f);
        v = v * v;
      }
      y[(size_t)(m0 + m) * N + n] = v;
    }
  }
}
