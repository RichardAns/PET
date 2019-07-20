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

struct Lor {
	int z1;
	int p1;
	int z2;
	int p2;
};

struct Ray {
	float3 a;
	float3 n;
	float lam1;
	float lam2;
};

__device__ int myAtomicAdd16(ushort *buf)
{
	ushort oldval = buf[0];
	ushort current = 0;
	do {
		current = oldval;
		oldval = atomicCAS(buf,current,current+1);
	} while (current != oldval);

	return (current == 65535) ? 1 : 0;
}

__device__ int ray_to_cyl(Ray &g,Lor &l,const cyl2D &c)
{
	//swim to cyl: slove quadratic
	float A = g.n.x*g.n.x + g.n.y*g.n.y;
	float B = g.a.x*g.n.x + g.a.y*g.n.y;  // factors of 2 ommited as they cancel
	float C = g.a.x*g.a.x + g.a.y*g.a.y - c.r*c.r;
	float D = B*B-A*C;
	float rad = sqrtf(D);
	g.lam1 = (-B+rad)/A;  // gamma1
	float z1 = g.a.z+g.lam1*g.n.z;
	g.lam2 = (-B-rad)/A;  // gamma2
	float z2 = g.a.z+g.lam2*g.n.z;  

	if(z1 >= 0.0f && z1 < c.len && z2 >= 0.0f && z2 < c.len){
		float x1  = g.a.x+g.lam1*g.n.x;
		float y1  = g.a.y+g.lam1*g.n.y;
		float phi = atan2(y1,x1)+cx::pi<float>;
		l.z1 = min(c.nz-1,  (int)(z1*c.zstep)   );
		l.p1 = min(c.nphi-1,(int)(phi*c.phistep));
		float x2  = g.a.x+g.lam2*g.n.x;
		float y2  = g.a.y+g.lam2*g.n.y;
		phi = atan2(y2,x2)+cx::pi<float>;
		l.z2 = min(c.nz-1,  (int)(z2*c.zstep)   );
		l.p2 = min(c.nphi-1,(int)(phi*c.phistep));
		return 1;
	}

	return 0;
}

template <typename S> __global__ void init_generator(long long seed,S *states)
{
	// minimal version
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed + id, 0, 0, &states[id]);  
	//curand_init(seed, id , 0, &states[id]);  
}

template <typename S> __global__ void cylhits(r_Ptr<uint> hits,r_Ptr<ushort> vfill,r_Ptr<double> ngood,cyl2D det, cyl3D roi,S *states,uint tries)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];
	//float4 a;
	//float4 n;
	Ray g;
	Lor lor;
	int good = 0;
	for(int k=0;k<tries;k++){
		float phi = cx::pi2<float>*curand_uniform(&state);
		float r = sqrtf(curand_uniform(&state))*roi.r;
		g.a.x = r*cosf(phi);
		g.a.y = r*sinf(phi);
		g.a.z = roi.len*curand_uniform(&state);
		if(1){
			int3 p;
			p.x = (int)((g.a.x+roi.r)*roi.xystep)/4;
			p.y = (int)((g.a.y+roi.r)*roi.xystep)/4;
			p.z = (int)(g.a.z*roi.zstep);
			p.z = min(roi.nz-1,p.z);
			//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
			if(p.x <0 || p.x >= roi.nxy || p.y < 0 || p.y >= roi.nxy || p.z < 0 || p.z >= roi.nz) printf("bad roi %d %d %d\n",p.x,p.y,p.z); 
			else myAtomicAdd16(&vfill[(p.z*roi.nxy+p.y)*roi.nxy+p.x]);  // but needs atomic!!
		    //else atomicAdd(&vfill[(p.z*roi.nxy+p.y)*roi.nxy+p.x],1); //needs atomic
																 //else vfill[(az*roi.nxy+ay)*roi.nxy+ax] += 1; //needs atomic
		}
		// generate isotropic back to back gammas
		phi = cx::pi2<float>*curand_uniform(&state);
		float theta = acosf(1.0f-2.0f*curand_uniform(&state));
		g.n.x = sinf(phi)*sinf(theta);
		g.n.y = cosf(phi)*sinf(theta);
		g.n.z = cosf(theta);
		if(ray_to_cyl(g,lor,det)){
			good++;
			atomicAdd(&hits[lor.p1*det.nz+lor.z1],1);
			atomicAdd(&hits[lor.p2*det.nz+lor.z2],1);

			if(0){
				int3 p;
				p.x = (int)((g.a.x+roi.r)*roi.xystep);
				p.y = (int)((g.a.y+roi.r)*roi.xystep);
				p.z = (int)(g.a.z*roi.zstep);
				p.z = min(roi.nz-1,p.z);
				//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
				if(p.x <0 || p.x >= roi.nxy || p.y < 0 || p.y >= roi.nxy || p.z < 0 || p.z >= roi.nz) printf("bad roi %d %d %d\n",p.x,p.y,p.z); 
				//else atomicAdd(&vfill[(p.z*roi.nxy+p.y)*roi.nxy+p.x],1); //needs atomic
				//else vfill[(p.z*roi.nxy+p.y)*roi.nxy+p.x] += 1; //needs atomic
			}

		}
		//swim to cyl: solve quadratic
		//float A = n.x*n.x + n.y*n.y;
		//float B = a.x*n.x + a.y*n.y;  // factors of 2 ommited as they cancel
		//float C = a.x*a.x + a.y*a.y - det.r*det.r;
		//float D = B*B-A*C;
		//float rad = sqrtf(D);
		//float lam1 = (-B+rad)/A;  // gamma1
		//float z1 = a.z+lam1*n.z;
		//float lam2 = (-B-rad)/A;
		//float z2 = a.z+lam2*n.z;  // gamma2
								  // accumulate hits
		//if(z1 >= 0.0f && z1 < det.len && z2 >= 0.0f && z2 < det.len){
		//	good++;
		//	float x1  = a.x+lam1*n.x;
		//	float y1  = a.y+lam1*n.y;
		//	phi = atan2f(y1,x1)+cx::pi<float>;
		//	az = min(det.nz-1,  (int)(z1*det.zstep)   );
		//	ay = min(det.nphi-1,(int)(phi*det.phistep));
		//	if(az < 0 || az >= det.nz || ay < 0 || ay >= det.nphi) printf("id %d L1 %d %d\n",id,az,ay);
		//	else atomicAdd(&hits[ay*det.nz+az],1);
		//	float x2  = a.x+lam2*n.x;
		//	float y2  = a.y+lam2*n.y;
		//	phi = atan2f(y2,x2)+cx::pi<float>;
		//	az = min(det.nz-1,  (int)(z2*det.zstep)   );
		//	ay = min(det.nphi-1,(int)(phi*det.phistep));
		//	if(az < 0 || az >= det.nz || ay < 0 || ay >= det.nphi) printf("id %d L2 %d %d\n",id,az,ay);
		//	else atomicAdd(&hits[ay*det.nz+az],1);
		//}
	}
	ngood[id] += good;
	states[id] = state;
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
	cyl2D det = {0.0f,0.0f,0.0f, 200.0f,400.0f ,256,256, 256.0f/(200.0f), 256.0f/cx::pi2<float> };  // outer cylinder	

	printf("cyl2d org (%.1f %.1f %.1f)  zlen %.1f radius %.1f nz %d nphi %d zstep %.3f phistep %.3f\n",
		         det.o.x,det.o.y,det.o.z,det.len,det.r,det.nz,det.nphi,det.zstep,det.phistep);
	
	//cyl3d origin len rad nz nxy zstep xystep
	cyl3D roi =  {0.0f,0.0f,0.0f, 200.0f,200.0f ,256,256, 256.0f/(200.0f), 256.0f/(2.0*200.0f)};  // inner cylinder
	printf("cyl2d org (%.1f %.1f %.1f)  zlen %.1f radius %.1f nz %d nxy %d zstep %.3f xystep %.3f\n",
		roi.o.x,roi.o.y,roi.o.z,roi.len,roi.r,roi.nz,roi.nxy,roi.zstep,roi.xystep);

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

	thrustHvec<uint> hits(det.nz*det.nphi);
	thrustHvec<uint> vfill(roi.nz*roi.nxy*roi.nxy);
	thrustHvec<ushort> sfill(roi.nz*roi.nxy*roi.nxy);
	thrustDvec<uint> dev_hits(det.nz*det.nphi);
	thrustDvec<uint> dev_vfill(roi.nz*roi.nxy*roi.nxy);
	thrustDvec<ushort> dev_sfill(roi.nz*roi.nxy*roi.nxy);

	thrustDvec<double> dev_good(size);
	cx::MYTimer tim;
	init_generator<<<blocks, threads>>>(seed,state.data().get());
	for(int k=0;k<passes;k++){
		//cylhits<<<blocks, threads >>>(dev_hits.data().get(), dev_vfill.data().get(), dev_good.data().get(), det, roi, state.data().get(), tries);
		cylhits<<<blocks, threads >>>(dev_hits.data().get(), dev_sfill.data().get(), dev_good.data().get(), det, roi, state.data().get(), tries);
	}
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
	tim.add();

	double all_good = thrust::reduce(dev_good.begin(), dev_good.end());
	ngen *= (long long)passes;
	double eff = 100.0*all_good/(double)ngen;
	printf("ngen %lld good %.0f eff %.3f%% time %.3f ms\n",ngen,all_good,eff,tim.time());

	hits = dev_hits;
	//vfill = dev_vfill;
	sfill = dev_sfill;
	//cx::write_raw("roi_int.raw",vfill.data(),roi.nz*roi.nxy*roi.nxy);
	cx::write_raw("roi_short.raw",sfill.data(),roi.nz*roi.nxy*roi.nxy);
	cx::write_raw("hits_int.raw",hits.data(),det.nz*det.nphi);


    return 0;
}