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
#define EXPORT(T,S) \
extern "C" __global__ void qwen38_fp8_mv_##S(const T*x,const unsigned char*w,const float*s,T*y,int M,int N,int K){mv(x,w,s,y,M,N,K);} \
extern "C" __global__ void qwen38_fp8_mm_##S(const T*x,const unsigned char*w,const float*s,T*y,int M,int N,int K){mm(x,w,s,y,M,N,K);}
EXPORT(__half,f16)
EXPORT(__nv_bfloat16,bf16)
extern "C" __global__ void qwen38_fp8_mv_f32(const float*x,const unsigned char*w,const float*s,float*y,int M,int N,int K){mv(x,w,s,y,M,N,K);}
