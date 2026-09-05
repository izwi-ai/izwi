#include <math.h>
#include <stdint.h>
#include <cub/block/block_scan.cuh>

__device__ bool before(float a,unsigned ai,float b,unsigned bi) {return a>b || (a==b && ai<bi);}
extern "C" __global__ void qwen38_greedy(const float*x,unsigned*out,int V) {
  int row=blockIdx.x,t=threadIdx.x;float best=-INFINITY;unsigned id=~0u;
  for(int i=t;i<V;i+=256) {float v=x[(size_t)row*V+i];if(isfinite(v) && before(v,i,best,id)){best=v;id=i;}}
  __shared__ float val[256];__shared__ unsigned ids[256];val[t]=best;ids[t]=id;__syncthreads();
  for(int d=128;d;d>>=1){if(t<d && before(val[t+d],ids[t+d],val[t],ids[t])){val[t]=val[t+d];ids[t]=ids[t+d];}__syncthreads();}
  if(t==0){out[row*2]=ids[0]==~0u?0:ids[0];out[row*2+1]=ids[0]!=~0u;}
}
extern "C" __global__ void qwen38_sampling_prepare(const float*x,const unsigned*counts,float*values,unsigned*ids,int V,int total,float temp,float rep,float presence,float frequency) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
  float v=x[i];unsigned count=counts[i];
  if(isfinite(v)) {if(count){if(rep>1.f)v=v>0?v/rep:v*rep;v-=presence+frequency*float(count);} if(temp>1e-5f)v/=temp;}
  values[i]=isfinite(v)?v:-INFINITY;ids[i]=i%V;
}
// Stable parallel merge. Each input binary-searches its unique output rank in
// the neighboring sorted run; exact ties use ascending token ID.
extern "C" __global__ void qwen38_sampling_merge(const float*src,const unsigned*si,float*dst,unsigned*di,int V,int total,int width) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
  int row=i/V,j=i%V,base=(j/(2*width))*(2*width),mid=min(base+width,V),end=min(base+2*width,V);
  bool left=j<mid;int lo=left?mid:base,hi=left?end:mid,start=lo;
  float v=src[i];unsigned id=si[i];
  while(lo<hi){int p=(lo+hi)/2;if(before(src[(size_t)row*V+p],si[(size_t)row*V+p],v,id))lo=p+1;else hi=p;}
  int rank=base+(j-(left?base:mid))+(lo-start);dst[(size_t)row*V+rank]=v;di[(size_t)row*V+rank]=id;
}
extern "C" __global__ void qwen38_sampling_probs(const float*sorted,const unsigned*ids,float*out,int V,int topk,float topp,float minp,int greedy) {
  int row=blockIdx.x,t=threadIdx.x;size_t base=(size_t)row*V;
  for(int i=t;i<V;i+=256)out[base+i]=0.f;
  __syncthreads();
  if(greedy){if(t==0 && isfinite(sorted[base]))out[base+ids[base]]=1.f;return;}
  int limit=topk>0?min(topk,V):V;float maxv=sorted[base];
  __shared__ float red[256],carry,kept;
  __shared__ unsigned cutoff;
  float sum=0;for(int i=t;i<limit;i+=256)sum+=isfinite(sorted[base+i])?expf(sorted[base+i]-maxv):0.f;
  red[t]=sum;__syncthreads();for(int d=128;d;d>>=1){if(t<d)red[t]+=red[t+d];__syncthreads();}
  float total=red[0];if(!(total>0.f) || !isfinite(total))return;
  if(t==0){carry=0;cutoff=limit;kept=0;}__syncthreads();
  typedef cub::BlockScan<float,256> Scan;__shared__ typename Scan::TempStorage scan;
  // Exact nucleus cutoff includes the first candidate crossing the threshold.
  for(int start=0;start<limit;start+=256){
    int i=start+t;float p=i<limit && isfinite(sorted[base+i])?expf(sorted[base+i]-maxv)/total:0.f;
    float prefix,aggregate;Scan(scan).InclusiveSum(p,prefix,aggregate);__syncthreads();
    if(i<limit && carry+prefix>=topp)atomicMin(&cutoff,(unsigned)(i+1));
    __syncthreads();if(t==0)carry+=aggregate;__syncthreads();
  }
  sum=0;for(int i=t;i<limit;i+=256){float v=sorted[base+i];float p=isfinite(v)?expf(v-maxv):0.f;if(i<cutoff && p>=minp && p>0.f){out[base+ids[base+i]]=p;sum+=p;}}
  red[t]=sum;__syncthreads();for(int d=128;d;d>>=1){if(t<d)red[t]+=red[t+d];__syncthreads();}
  float norm=red[0];if(norm>0.f)for(int i=t;i<V;i+=256)out[base+i]/=norm;
}

// Sampling and rejection sampling share the same parallel inverse-CDF scan.
// Meta is [uniform] for sampling; [proposal, acceptance draw, residual draw] for
// verification. All RNG draws are caller-owned so transaction rollback is exact.
template<bool Verify> __device__ void select(const float*p,const float*q,const float*meta,const unsigned*order,unsigned*out,int V) {
  int row=blockIdx.x,t=threadIdx.x;size_t base=(size_t)row*V;
  __shared__ float red[256],carry;__shared__ unsigned bad[256],choice,last;__shared__ int accept;
  float sum=0;unsigned invalid=0;
  for(int i=t;i<V;i+=256){float v=p[base+i];invalid|=!isfinite(v)||v<0.f;if(Verify){float z=q[base+i];invalid|=!isfinite(z)||z<0.f;}sum+=v;}
  red[t]=sum;bad[t]=invalid;__syncthreads();for(int d=128;d;d>>=1){if(t<d){red[t]+=red[t+d];bad[t]|=bad[t+d];}__syncthreads();}
  if(t==0){
    accept=0;choice=~0u;last=0;carry=0;
    float u=meta[row*(Verify?3:1)+(Verify?1:0)];
    if(!(u>=0.f&&u<1.f)||bad[0]||!(red[0]>0.f)||!isfinite(red[0]))accept=-1;
    if(Verify){float d=meta[row*3];float r=meta[row*3+2];
      if(!(d>=0.f && d<V && d==floorf(d))||!(r>=0.f&&r<1.f))accept=-1;
      else if(accept==0){unsigned token=(unsigned)d;float qd=q[base+token];if(!(qd>0.f))accept=-1;else if(u<fminf(1.f,p[base+token]/qd)){accept=1;choice=token;}}
    }
  }__syncthreads();
  if(accept!=0){if(t==0){if(Verify){out[row*3]=accept==1;out[row*3+1]=accept==1?choice:0;out[row*3+2]=accept==1;}else{out[row*2]=0;out[row*2+1]=0;}}return;}
  sum=0;for(int i=t;i<V;i+=256){float v=Verify?fmaxf(p[base+i]-q[base+i],0.f):p[base+i];sum+=v;}
  red[t]=sum;__syncthreads();for(int d=128;d;d>>=1){if(t<d)red[t]+=red[t+d];__syncthreads();}
  float total=red[0],draw=meta[row*(Verify?3:1)+(Verify?2:0)]*total;
  typedef cub::BlockScan<float,256> Scan;__shared__ typename Scan::TempStorage scan;
  for(int start=0;start<V;start+=256){int i=start+t;unsigned id=i<V?order[base+i]:0;float v=i<V?(Verify?fmaxf(p[base+id]-q[base+id],0.f):p[base+id]):0.f;float prefix,aggregate;
    Scan(scan).InclusiveSum(v,prefix,aggregate);__syncthreads();
    if(v>0.f){atomicMax(&last,(unsigned)i);if(draw<carry+prefix)atomicMin(&choice,(unsigned)i);}
    __syncthreads();if(t==0)carry+=aggregate;__syncthreads();
  }
  if(t==0){bool valid=total>0.f&&isfinite(total);unsigned token=order[base+(choice==~0u?last:choice)];if(Verify){out[row*3]=0;out[row*3+1]=token;out[row*3+2]=valid;}else{out[row*2]=token;out[row*2+1]=valid;}}
}
extern "C" __global__ void qwen38_sample(const float*p,const float*u,const unsigned*order,unsigned*out,int V){select<false>(p,nullptr,u,order,out,V);}
extern "C" __global__ void qwen38_verify(const float*p,const float*q,const float*meta,const unsigned*order,unsigned*out,int V){select<true>(p,q,meta,order,out,V);}

extern "C" __global__ void qwen38_sampling_order(const float*p,float*sorted,unsigned*ids,int V,int total){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total){sorted[i]=isfinite(p[i])?p[i]:-INFINITY;ids[i]=i%V;}}
