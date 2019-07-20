// start very simple cylindrical geom
#pragma warning( disable : 4267)   // size_t int mismatch 
#pragma warning( disable : 4244)   // thrust::reduce int mismatch

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "helper_cuda.h"
#include <curand_kernel.h> 

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

//#include "mystuff.h"
#include "cx.h"
#include "timers.h"
//cyl2d origin len rad nz nphi zstep phistep
struct cyl2D {
	float3 o; // origin
	float len;  // length
	float r;  // radius
	int nz;   // len bins
	int nphi; // phi bins on surface
	float zstep;
	float phistep;
};
//cyl3d origin len rad nz nxy zstep xystep
struct cyl3D {
	float3 o;  // origin
	float len; // length
	float r;   // radius
	int nz;    // z bins
	int nxy;   // x-y bins
	float zstep;
	float xystep;
};

template <typename S> __global__ void init_generator(long long seed,S *states)
{
	// minimal version
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed + id, 0, 0, &states[id]);  
	//curand_init(seed, id , 0, &states[id]);  
}

template <typename S> __global__ void cylhits(r_Ptr<float> hits,r_Ptr<float> vfill,cyl2D out, cyl3D roi,S *states,uint tries)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];
	float4 a;
	float4 n;
	for(int k=0;k<tries;k++){
		float phi = cx::pi2<float>*curand_uniform(&state);
		float r = sqrtf(curand_uniform(&state))*roi.r;
		a.x = r*cosf(phi);
		a.y = r*sinf(phi);
		a.z = roi.len*curand_uniform(&state);
		int ax = (int)((a.x+roi.r)*roi.xystep);
		int ay = (int)((a.y+roi.r)*roi.xystep);
		int az = (int)(a.z*roi.zstep);
		az = min(roi.nz-1,az);
		//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
		//if(ax <0 || ax >= roi.nxy || ay < 0 || ay >= roi.nxy || az < 0 || az >= roi.nz) printf("bad roi %d %d %d\n",ax,ay,az); 
		//else atomicAdd(&vfill[(az*roi.nxy+ay)*roi.nxy+ax],1.0f); //needs atomic
	
		// generate isotropic back to back gammas
		phi = cx::pi2<float>*curand_uniform(&state);
		float theta = acosf(1.0f-2.0f*curand_uniform(&state));
		n.x = sinf(phi)*sinf(theta);
		n.y = cosf(phi)*sinf(theta);
		n.z = cosf(theta);

		//swim to cyl: slove quadratic
		float A = n.x*n.x + n.y*n.y;
		float B = a.x*n.x + a.y*n.y;  // factors of 2 ommited as they cancel
		float C = a.x*a.x + a.y*a.y - out.r*out.r;
		float D = B*B-A*C;
		float rad = sqrtf(D);
		float lam1 = (-B+rad)/A;  // gamma1
		float z1 = a.z+lam1*n.z;
		float lam2 = (-B-rad)/A;
		float z2 = a.z+lam2*n.z;  // gamma2
		// accumulate hits
		if(z1 >= 0.0f && z1 < out.len && z2 >= 0.0f && z2 < out.len){
			float x1  = a.x+lam1*n.x;
			float y1  = a.y+lam1*n.y;
			phi = atan2f(y1,x1)+cx::pi<float>;
			az = min(out.nz-1,  (int)(z1*out.zstep)   );
			ay = min(out.nphi-1,(int)(phi*out.phistep));
			if(az < 0 || az >= out.nz || ay < 0 || ay >= out.nphi) printf("id %d L1 %d %d\n",id,az,ay);
			else atomicAdd(&hits[ay*out.nz+az],1.0f);
			float x2  = a.x+lam2*n.x;
			float y2  = a.y+lam2*n.y;
			phi = atan2f(y2,x2)+cx::pi<float>;
			az = min(out.nz-1,  (int)(z2*out.zstep)   );
			ay = min(out.nphi-1,(int)(phi*out.phistep));
			if(az < 0 || az >= out.nz || ay < 0 || ay >= out.nphi) printf("id %d L2 %d %d\n",id,az,ay);
			else atomicAdd(&hits[ay*out.nz+az],1.0f);
		}
	}

}

template <typename S> __global__ void cylhits_int(r_Ptr<uint> hits,r_Ptr<uint> vfill,cyl2D out, cyl3D roi,S *states,uint tries)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];
	float4 a;
	float4 n;
	for(int k=0;k<tries;k++){
		float phi = cx::pi2<float>*curand_uniform(&state);
		float r = sqrtf(curand_uniform(&state))*roi.r;
		a.x = r*cosf(phi);
		a.y = r*sinf(phi);
		a.z = roi.len*curand_uniform(&state);
		int ax = (int)((a.x+roi.r)*roi.xystep);
		int ay = (int)((a.y+roi.r)*roi.xystep);
		int az = (int)(a.z*roi.zstep);
		az = min(roi.nz-1,az);
		//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
		//if(ax <0 || ax >= roi.nxy || ay < 0 || ay >= roi.nxy || az < 0 || az >= roi.nz) printf("bad roi %d %d %d\n",ax,ay,az); 
		//else atomicAdd(&vfill[(az*roi.nxy+ay)*roi.nxy+ax],1.0f); //needs atomic

		// generate isotropic back to back gammas
		phi = cx::pi2<float>*curand_uniform(&state);
		float theta = acosf(1.0f-2.0f*curand_uniform(&state));
		n.x = sinf(phi)*sinf(theta);
		n.y = cosf(phi)*sinf(theta);
		n.z = cosf(theta);

		//swim to cyl: slove quadratic
		float A = n.x*n.x + n.y*n.y;
		float B = a.x*n.x + a.y*n.y;  // factors of 2 ommited as they cancel
		float C = a.x*a.x + a.y*a.y - out.r*out.r;
		float D = B*B-A*C;
		float rad = sqrtf(D);
		float lam1 = (-B+rad)/A;  // gamma1
		float z1 = a.z+lam1*n.z;
		float lam2 = (-B-rad)/A;
		float z2 = a.z+lam2*n.z;  // gamma2
								  // accumulate hits
		if(z1 >= 0.0f && z1 < out.len && z2 >= 0.0f && z2 < out.len){
			float x1  = a.x+lam1*n.x;
			float y1  = a.y+lam1*n.y;
			phi = atan2f(y1,x1)+cx::pi<float>;
			az = min(out.nz-1,  (int)(z1*out.zstep)   );
			ay = min(out.nphi-1,(int)(phi*out.phistep));
			if(az < 0 || az >= out.nz || ay < 0 || ay >= out.nphi) printf("id %d L1 %d %d\n",id,az,ay);
			else atomicAdd(&hits[ay*out.nz+az],1);
			float x2  = a.x+lam2*n.x;
			float y2  = a.y+lam2*n.y;
			phi = atan2f(y2,x2)+cx::pi<float>;
			az = min(out.nz-1,  (int)(z2*out.zstep)   );
			ay = min(out.nphi-1,(int)(phi*out.phistep));
			if(az < 0 || az >= out.nz || ay < 0 || ay >= out.nphi) printf("id %d L2 %d %d\n",id,az,ay);
			else atomicAdd(&hits[ay*out.nz+az],1);
		}
	}

}

int main(int argc,char *argv[])
{
	printf("f pi is %.10f\n",cx::pi<float>);
	printf("d pi is %.10f\n",cx::pi<double>);

	if(argc < 2){
		printf("usage simple threads block ngen seed R L r l\n");
		return 0;
	}
	//cyl2d origin len rad nz nphi zstep phistep
	cyl2D Cout = {0.0f,0.0f,0.0f, 200.0f,400.0f ,256,256, 256.0f/(200.0f), 256.0f/cx::pi2<float> };  // outer cylinder	

	printf("cyl2d org (%.1f %.1f %.1f)  zlen %.1f radius %.1f nz %d nphi %d zstep %.3f phistep %.3f\n",
		         Cout.o.x,Cout.o.y,Cout.o.z,Cout.len,Cout.r,Cout.nz,Cout.nphi,Cout.zstep,Cout.phistep);
	
	//cyl3d origin len rad nz nxy zstep xystep
	cyl3D Cin =  {0.0f,0.0f,0.0f, 200.0f,200.0f ,256,256, 256.0f/(200.0f), 256.0f/(2.0*200.0f)};  // inner cylinder
	printf("cyl2d org (%.1f %.1f %.1f)  zlen %.1f radius %.1f nz %d nxy %d zstep %.3f xystep %.3f\n",
		Cin.o.x,Cin.o.y,Cin.o.z,Cin.len,Cin.r,Cin.nz,Cin.nxy,Cin.zstep,Cin.xystep);

	uint threads = 256; if(argc>1) threads = atoi(argv[1]);
	uint blocks = 1024; if(argc>2) blocks  = atoi(argv[2]);

	uint size = blocks*threads;
	int passes = 1;
	long long  ngen = 1000000;
	int ndo = atoi(argv[3]);
	if(ndo <1000) ngen *= (long long)ndo;
	else {
		passes = ndo/1000;
		ngen *= 1000ll;
	}
	uint tries = (ngen+size-1)/size;
	ngen = (long long)tries*(long long)size;

	std::random_device rd;
	long long seed = rd(); if (argc > 4) seed = atoi(argv[4]);

	thrustDvec<curandState> state(size);  // this for curand_states
	curandState *state_ptr = state.data().get();

	thrustHvec<float> hits(Cout.nz*Cout.nphi);
	thrustHvec<float> roi(Cin.nz*Cin.nxy*Cin.nxy);
	thrustDvec<float> dev_hits(Cout.nz*Cout.nphi);
	thrustDvec<float> dev_roi(Cin.nz*Cin.nxy*Cin.nxy);

	thrustHvec<uint> hits_int(Cout.nz*Cout.nphi);
	thrustHvec<uint> roi_int(Cin.nz*Cin.nxy*Cin.nxy);
	thrustDvec<uint> dev_hits_int(Cout.nz*Cout.nphi);
	thrustDvec<uint> dev_roi_int(Cin.nz*Cin.nxy*Cin.nxy);

	cx::MYTimer tim;
	init_generator<<<blocks, threads>>>(seed,state.data().get());
	for(int k=0;k<passes;k++){
		//cylhits<<<blocks, threads >>>(dev_hits.data().get(),dev_roi.data().get(),Cout,Cin,state.data().get(),tries);
		cylhits_int<<<blocks, threads >>>(dev_hits_int.data().get(),dev_roi_int.data().get(),Cout,Cin,state.data().get(),tries);
	}
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
	tim.add();

	ngen *= (long long)passes;
	printf("ngen %lld time %.3f ms\n",ngen,tim.time());
	//hits = dev_hits;
	//roi = dev_roi;
	//cx::write_raw("roi.raw",roi.data(),Cin.nz*Cin.nxy*Cin.nxy);
	//cx::write_raw("hits.raw",hits.data(),Cout.nz*Cout.nphi);
	hits_int = dev_hits_int;
	roi_int = dev_roi_int;
	cx::write_raw("roi_int.raw",roi_int.data(),Cin.nz*Cin.nxy*Cin.nxy);
	cx::write_raw("hits_int.raw",hits_int.data(),Cout.nz*Cout.nphi);


    return 0;
}