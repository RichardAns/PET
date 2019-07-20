 // start very simple cylindrical geom
#pragma warning( disable : 4267)   // size_t int mismatch 
#pragma warning( disable : 4244)   // thrust::reduce int mismatch

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "helper_cuda.h"
#include <curand_kernel.h> 

#include <stdio.h>
#include <stdlib.h>
#include <random>

//#include "mystuff.h"
#include "cx.h"
#include "timers.h"
#include "scanner.h"

// for clockwise phi rotation with phi= 0 at 12 oclock (+ve y axis)
// use with y =cos(phi) x = sin(phi)
template <typename T> __host__ __device__ T myatan2(T x,T y){
	T angle = atan2(x,y);
	if(angle <0) angle += cx::pi2<T>;
	return angle;
}

__host__ __device__ int phi2cry(float phi)
{
	while(phi < 0.0f) phi += cx::pi2<float>;
	while(phi >= cx::pi2<float>) phi -= cx::pi2<float>;
	return (int)( phi*cryStep );
}

__host__ __device__ float cry2phi(int cry)
{
	while(cry < 0) cry += cryNum;
	while(cry >= cryNum) cry -= cryNum;
	return ( (float)cry+0.5f )*phiStep;  // phi at crystal centre
}

// indexing modulo Range, assumes i is within interval
template <int Range> __host__ __device__ int cyc_sub(cint i,int step) {
	return i >= step ? i-step : i-step+Range;
}

template <int Range> __host__ __device__ int cyc_dec(cint i) {
	return i >= 1 ? i-1 : Range-1;
}

template <int Range> __host__ __device__ int cyc_add(cint i,int step) {
	return i+step < Range  ? i+step : i+step-Range;
}

template <int Range> __host__ __device__ int cyc_inc(cint i) {
	return i+1 < Range  ? i+1 : 0;
}

__device__ int myatomicAdd16(ushort *buf)
{
	ushort oldval = buf[0];
	ushort current = 0;
	do {
		current = oldval;
		oldval = atomicCAS(buf,current,current+1);
	} while (current != oldval);

	return (current == 65535) ? 1 : 0;  // overflow bit ?
}

// TODO allow any length of cylinder so can use for SM generation in long detector
// and phantom generation in short detector. 
// Add detlen parameter to save time with eraly return if out of detector.
__device__ int ray_to_cyl(Ray &g,Lor &l,float length)  //15/6/19 added length argument  
{
	//swim to cyl: solve quadratic
	float A = g.n.x*g.n.x + g.n.y*g.n.y;
	float B = g.a.x*g.n.x + g.a.y*g.n.y;  // factors of 2 ommited as they cancel
	float C = g.a.x*g.a.x + g.a.y*g.a.y - detRadius*detRadius;
	float D = B*B-A*C;
	float rad = sqrtf(D);
	g.lam1 = (-B+rad)/A;  // gamma1
	float z1 = g.a.z+g.lam1*g.n.z;
	g.lam2 = (-B-rad)/A;  // gamma2
	float z2 = g.a.z+g.lam2*g.n.z;  

	if(z1 >= 0.0f && z1 < length && z2 >= 0.0f && z2 < length && abs(z2-z1) < detLen ){ // same zdiff short and long detectors
		float x1  = g.a.x+g.lam1*g.n.x;
		float y1  = g.a.y+g.lam1*g.n.y;
		float phi = myatan2(x1,y1);
		l.z1 =  (int)(z1*detStep); //min(zLongNum-1,(int)(z1*detStep)); 
		l.c1 =  phi2cry(phi);      // min(cryNum-1,(int)(phi*cryStep)); 
		float x2  = g.a.x+g.lam2*g.n.x;
		float y2  = g.a.y+g.lam2*g.n.y;
		phi = myatan2(x2,y2);
		l.z2 = min(zLongNum-1,  (int)(z2*detStep)   );  // (int)(z2*detStep); 
		l.c2 = min(cryNum-1,(int)(phi*cryStep));        //phi2cry(phi);  

		if(l.z1 > l.z2){
			cx::swap(l.z1,l.z2);
			cx::swap(l.c1,l.c2);
		}
		else if(l.z1==l.z2 && l.c1 > l.c2) cx::swap(l.c1,l.c2);
		return 1;
	}

	return 0;
}

template <typename T> __global__ void clear_vec(T *vec,size_t count)
{
	size_t id = threadIdx.x + blockIdx.x*blockDim.x;
	while(id < count) {
		vec[id] = 0;
		id += gridDim.x*blockDim.x;
	}
}

template <typename S> __global__ void init_generator(long long seed,S *states)
{
	// minimal version
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed + id, 0, 0, &states[id]);  
	//curand_init(seed, id , 0, &states[id]);  
}

__device__ __host__ float3 roi2xyz(Roi &v)
{
	float3 a;
	float phi = 0.5f*(v.phi.y+v.phi.x);   // averge phi
	float rxy = 0.5f*(v.r.y+v.r.x);       // distance from z axis
	a.x = rxy*sinf(phi);                  // centre of voxel
	a.y = rxy*cosf(phi);
	a.z = 0.5f*(v.z.y+v.z.x);           // average z
	printf("Roi  phi (%7.3f %7.3f) r (%7.1f %7.1f) z (%7.1f %7.1f) a (%8.2f %8.2f %8.2f)\n",v.phi.x,v.phi.y,v.r.x,v.r.y,v.z.x,v.z.y,a.x,a.y,a.z);
	return a;
}

__global__ void find_spot(r_Ptr<uint> map,r_Ptr<uint> spot,Roi vox)
{
	float c_id = threadIdx.x; // left hand crystal
	float z_id = blockIdx.x;  // left handz
	if(c_id >= cryNum) return;  //    sanity checks
	if(z_id >= zNum) return;
	Ray r; 
	float phi = (cx::pi2<>/cryNum)*(c_id+0.5f);
	r.a.x = detRadius*sinf(phi);  // centre of crystal
	r.a.y = detRadius*cosf(phi);  // phi = 0 along y axis increasing clockwise
	r.a.z = crySize*(z_id+0.5f);
	
	phi = 0.5f*(vox.phi.y+vox.phi.x);   // averge phi
	float rxy = 0.5f*(vox.r.y+vox.r.x); // distance from z axis
	r.n.x = rxy*sinf(phi);              // centre of voxel
	r.n.y = rxy*cosf(phi);
	r.n.z = 0.5f*(vox.z.y+vox.z.x);     // average z
	// find lam such that a+ lam*b hits cyl
	r.lam2 = -2.0f*(r.a.x*r.n.x + r.a.y*r.n.y-detRadius*detRadius)/((r.n.x-r.a.x)*(r.n.x-r.a.x) + (r.n.y-r.a.y)*(r.n.y-r.a.y));
	float3 b; // centre of crytal cluster for this lor family
	b.x = r.a.x+r.lam2*(r.n.x-r.a.x);
	b.y = r.a.y+r.lam2*(r.n.y-r.a.y);
	b.z = r.a.z+r.lam2*(r.n.z-r.a.z);
	phi = myatan2(b.x,b.y);
	int c = phi2cry(phi);
	int z = b.z/crySize - zNum + 1;   //detRings/2;

	
    z = min(zNum-1, z);  // spot z in valid range (does not have to be)
	z = max(0, z);
	// copy hits to spot map
	size_t m_slice = (z_id*cryNum+c_id)*mapSlice;
	size_t s_slice = (z_id*cryNum+c_id)*spotNphi*spotNz;

	int sz = max(0,z-(2*spotNz/3));
	for(int iz=0;iz<spotNz;iz++){
		int sc = cyc_sub<cryNum>(c,spotNphi/2);
		for(int ic = 0;ic<spotNphi;ic++){
			uint val =map[m_slice+sz*cryNum+sc];
			spot[s_slice+iz*spotNphi+ic] =val;
			sc = cyc_inc<cryNum>(sc);
		}
		sz++;
		if (sz >= zNum) break;
	}

	spot[s_slice] = max(0,z-(2*spotNz/3));
	spot[s_slice+spotNphi] = cyc_sub<cryNum>(c,spotNphi/2);
	//map[index + (z*cryNum+c)]    = 0;  // mark centre for debug


	//if(c_id==205 && z_id==36) printf("c %3d z %2d rxy %.3f ,lam2 %.6f a (%.3f %.3f %.3f) n (%.3f %.3f %.3f) b (%.3f %.3f %.3f)\n",
	//	          c,z,rxy,r.lam2, r.a.x,r.a.y,r.a.z, r.n.x,r.n.y,r.n.z, b.x,b.y,b.z);
	//z = max(0,z); 
	//map[index + (48*cryNum+200)] = c;
	//map[index + (49*cryNum+200)] = z;

}

template <typename S> __global__ void cylhits(r_Ptr<uint> hits,r_Ptr<uint> map,r_Ptr<ushort> vfill,r_Ptr<double> ngood, Roi roi,S *states,uint tries)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];
	//float4 a;
	//float4 n;
	Ray g;
	Lor lor;
	int good = 0;
	float r1sq = roi.r.x*roi.r.x;
	float r2sq = roi.r.y*roi.r.y-r1sq;
	float dphi = roi.phi.y-roi.phi.x;
	float dz =   roi.z.y-roi.z.x;
	for(uint k=0;k<tries;k++){
		// generate point in roi
		float phi = roi.phi.x + dphi*curand_uniform(&state);
		float r =   sqrtf(r1sq +r2sq*curand_uniform(&state));			
		g.a.x = r*sinf(phi);
		g.a.y = r*cosf(phi);
		g.a.z = roi.z.x + dz*curand_uniform(&state);
		if(0){  // save generated decay points debug only
			int3 p;
			p.x = (int)((g.a.x+roiRadius)*voxStep);
			p.y = (int)((g.a.y+roiRadius)*voxStep);
			p.z = (int)(g.a.z*detStep);
			p.z = min(zLongNum-1,p.z);
			//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
			if(p.x <0 || p.x >= voxNum || p.y < 0 || p.y >= voxNum || p.z < 0 || p.z >= zLongNum) printf("bad roi %d %d %d\n",p.x,p.y,p.z); 
			else myatomicAdd16(&vfill[(p.z*voxNum+p.y)*voxNum+p.x]); 
		    //else atomicAdd(&vfill[(p.z*roi.nxy+p.y)*roi.nxy+p.x],1); 
			//else vfill[(az*roi.nxy+ay)*roi.nxy+ax] += 1; 
		}
		// generate isotropic back to back gammas
		phi = cx::pi2<float>*curand_uniform(&state);
		float theta = acosf(1.0f-2.0f*curand_uniform(&state));
		g.n.x = sinf(phi)*sinf(theta);
		g.n.y = cosf(phi)*sinf(theta);
		g.n.z = cosf(theta);
		if(ray_to_cyl(g,lor,detLongLen)){
			good++;
			//if(id==0 && good<100) printf("%3d (%3d %3d)-(%3d %3d) phi %7.2f theta %7.2f\n",good,lor.z1,lor.c1,lor.z2,lor.c2,phi*180.0f/cx::pi<float>,theta*180.0f/cx::pi<float> );
			//if(lor.c1==205 && lor.z1==36) {
			//	printf("%3d lor (%3d %3d)-(%3d %3d) phi %7.2f theta %7.2f g.a (%6.1f %6.1f %6.1f) g.n (%6.3f %6.3f %6.3f)\n",good,lor.z1,lor.c1,lor.z2,lor.c2,phi*180.0f/cx::pi<float>,theta*180.0f/cx::pi<float>,g.a.x,g.a.y,g.a.z,g.n.x,g.n.y,g.n.z );
			//}
			if(0){  // more debug
				atomicAdd(&hits[lor.c1*zLongNum+lor.z1],1);  // all lors in single plane
				atomicAdd(&hits[lor.c2*zLongNum+lor.z2],1);
				int3 p;
				p.x = (int)((g.a.x+detRadius)*voxStep);
				p.y = (int)((g.a.y+detRadius)*voxStep);
				p.z = (int)(g.a.z*detStep);
				p.z = min(zLongNum-1,p.z);
				//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
				if(p.x <0 || p.x >= voxNum || p.y < 0 || p.y >= voxNum || p.z < 0 || p.z >= zLongNum) printf("bad roi %d %d %d\n",p.x,p.y,p.z); 
				//else atomicAdd(&vfill[(p.z*roi.nxy+p.y)*roi.nxy+p.x],1); //needs atomic
				//else vfill[(p.z*roi.nxy+p.y)*roi.nxy+p.x] += 1; //needs atomic
			}
			if(1){
				if(lor.z2<zNum-1) printf("unexpected z2 = %d\n",lor.z2);
				uint lor2sub = max(0,lor.z2-zNum+1);
				if(lor.z1 <1) printf("unexpected z1 = %d\n",lor.z1);
				uint lor1sub = max(0,lor.z1-1);
				size_t index = (lor1sub*cryNum+lor.c1)*mapSlice+(lor2sub*cryNum+lor.c2);
				//size_t index = (lor.z1*cryNum+lor.c1)*mapSlice+(lor.z2*cryNum+lor.c2);
				atomicAdd(&map[index],1);
				//int overflow = myatomicAdd16(&map[index]);
				//if(overflow)   myatomicAdd16(&map[mapSlice*mapSlice+index]);
			}

		}
	}
	ngood[id] += good;
	states[id] = state;


}

//NB this can be called with either z1 or (z2-z1) as argument
//   steps in the other variable will then be adjacent in memory
//   Using (z2-z1) are argument turns out to be a bit faster.
__device__ int zdz_slice(int z)
{
	return detZdZNum - (zNum-z)*(zNum-z+1)/2;
}

template <typename S> __global__ void phantom(r_Ptr<uint> map,r_Ptr<uint> vfill,r_Ptr<uint> vfill2, PRoi roi, r_Ptr<double> ngood, S *states,uint tries,int dovol)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	S state = states[id];

	Ray g;
	Lor lor;
	int good = 0;
	float r1sq = roi.r.x*roi.r.x;
	float r2sq = roi.r.y*roi.r.y-r1sq;
	float dphi = roi.phi.y-roi.phi.x;
	float dz =   roi.z.y-roi.z.x;
	int3 p;
	for(uint k=0;k<tries;k++){
		// generate decay point at a
		float phi = roi.phi.x + dphi*curand_uniform(&state);
		float r =   sqrtf(r1sq +r2sq*curand_uniform(&state));			
		g.a.x = r*sinf(phi) +roi.o.x;
		g.a.y = r*cosf(phi) +roi.o.y;
		g.a.z = roi.z.x + dz*curand_uniform(&state) +roi.o.z;
		if(dovol){  // save generated decay points debug only
			p.x = (int)((g.a.x+roiRadius)*voxStep);
			p.y = (int)((g.a.y+roiRadius)*voxStep);
			p.z = (int)(g.a.z*detStep);
			p.z = min(zNum-1,p.z);
			//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
			if(p.x <0 || p.x >= voxNum || p.y < 0 || p.y >= voxNum || p.z < 0 || p.z >= zNum) printf("bad roi %d %d %d\n",p.x,p.y,p.z); 
			//else myatomicAdd16(&vfill[(p.z*voxNum+p.y)*voxNum+p.x]); 
			else atomicAdd(&vfill[(p.z*voxNum+p.y)*voxNum+p.x],1); 
			//else vfill[(az*roi.nxy+ay)*roi.nxy+ax] += 1; 
		}

		// generate isotropic back to back gammas int +/- n direction
		phi = cx::pi2<float>*curand_uniform(&state);
		float theta = acosf(1.0f-2.0f*curand_uniform(&state));
		g.n.x = sinf(phi)*sinf(theta);
		g.n.y = cosf(phi)*sinf(theta);
		g.n.z = cosf(theta);
		if(ray_to_cyl(g,lor,detLen)){
			int cdiff = abs(lor.c1-lor.c2); 
			if(cdiff >= cryDiffMin && cdiff <=cryDiffMax){
				good++;
				if(dovol)atomicAdd(&vfill2[(p.z*voxNum+p.y)*voxNum+p.x],1); 
				uint lor2fix = max(0,lor.z2);  lor2fix = min(lor2fix,63);
				uint lor1fix = max(0,lor.z1);  lor1fix = min(lor1fix,63);
				//size_t index = (lor1fix*cryNum+lor.c1)*mapSlice+(lor2fix*cryNum+lor.c2); //standard map
				//size_t zdz = detZdZNum - (zNum-lor.z1)*(zNum-lor.z1+1)/2 +lor.z2-lor.z1;
				//size_t cdc = cyc_sub<cryNum>(lor.c2,lor.c1) - cryDiffMin;
				//size_t index =  zdz*cryCdCNum+cdc*cryNum+lor.c1;
				
				uint zdz =    zdz_slice(lor.z2-lor.z1)+lor.z1;
				uint cdc =    cyc_sub<cryNum>(lor.c2,lor.c1) - cryDiffMin;
				uint index =  zdz*cryCdCNum+cdc*cryNum+lor.c1;

				atomicAdd(&map[index],1);
			}
		}
	} // done
	ngood[id] += good;
	states[id] = state;
}

int do_phantom(int argc,char *argv[])
{
	if(atoi(argv[1]) != 2) return 1;
	uint threads = atoi(argv[2]);
	uint blocks =  atoi(argv[3]);

	uint size = blocks*threads;
	int passes = 1;
	long long  ngen = 1000000;
	int ndo = atoi(argv[4]);
	if(ndo <1000) ngen *= (long long)ndo;
	else {
		passes = ndo/1000;
		ngen *= 1000ll;
	}
	uint tries = (ngen+size-1)/size;
	ngen = (long long)tries*(long long)size;
	long long ngen_all = ngen*(long long)passes;
	std::random_device rd;
	long long seed = rd(); if (atoi(argv[5]) > 0) seed = atoi(argv[5]);


	PRoi roi;
	// outfile in argv[6]
	roi.r.x =  atof(argv[7]);
	roi.r.y =  atof(argv[8]);   
	roi.phi.x = atof(argv[9])*cx::pi<float>/180.0f;
	roi.phi.y = atof(argv[10])*cx::pi<float>/180.0f;
	roi.z.x = atof(argv[11]);  
	roi.z.y = atof(argv[12]); 


	roi.o.x = 0.0f;  if(argc >13) roi.o.x = atof(argv[13]);
	roi.o.y = 0.0f;  if(argc >14) roi.o.y = atof(argv[14]);
	roi.o.z = 0.0f;  if(argc >15) roi.o.z = atof(argv[15]);

	int dovol = 0;   if(argc >16) dovol= atof(argv[16]);
	int savemap = 0; if(argc >17) savemap = atof(argv[17]);

	printf("Phantom r (%.1f %.1f) p (%.3f %.3f) z (%.1f %.1f) o (%.1f %.1f %.1f)\n",
		roi.r.x,roi.r.y, roi.phi.x,roi.phi.y, roi.z.x,roi.z.y, roi.o.x,roi.o.y,roi.o.z);

	// use XORWOW
	thrustDvec<curandState> state(size);  // this for curand_states

	uint vsize = zNum*voxNum*voxNum;
	uint msize = mapSlice*mapSlice;
	uint zsize = cryNum*cryDiffNum*detZdZNum;
	thrustHvec<uint>       vfill(vsize);
	thrustDvec<uint>   dev_vfill(vsize);
	thrustHvec<uint>       vfill2(vsize);
	thrustDvec<uint>   dev_vfill2(vsize);
	thrustHvec<uint>       map(msize);
	thrustDvec<uint>   dev_map(msize);
	thrustDvec<double> dev_good(size);
	thrustHvec<uint>       zdzmap(zsize);
	thrustDvec<uint>   dev_zdzmap(zsize);

	// if file exists we append
	if(cx::file_can_be_opened(argv[6])){
		if(cx::read_raw(argv[6],zdzmap.data(),zsize) != 0) for(uint k=0;k<zsize;k++) zdzmap[k] = 0;
		else dev_zdzmap = zdzmap;
	}

	cx::MYTimer tim;
	init_generator<<<blocks, threads>>>(seed,state.data().get());
	for(int k=0;k<passes;k++){
		//phantom<<<blocks, threads >>>(dev_map.data().get(), dev_vfill.data().get(), roi, dev_good.data().get(), state.data().get(), tries,dovol);
		phantom<<<blocks, threads >>>(dev_zdzmap.data().get(), dev_vfill.data().get(), dev_vfill2.data().get(), roi, dev_good.data().get(), state.data().get(), tries,dovol);

	}
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
	tim.add();
	double all_good = thrust::reduce(dev_good.begin(), dev_good.end());
	double eff = 100.0*all_good/(double)ngen_all;

	printf("Phantom ngen %lld good %.0f eff %.3f%% time %.3f ms\n",ngen_all,all_good,eff,tim.time());

	if(dovol){
		vfill = dev_vfill;
		cx::write_raw("phant_roi_all.raw",vfill.data(),zNum*voxNum*voxNum);
		vfill = dev_vfill2;
		cx::write_raw("phant_roi_good.raw",vfill.data(),zNum*voxNum*voxNum);
	}
	if(savemap){
		//map = dev_map;
		//cx::write_raw("phant_map_int.raw",map.data(),mapSlice*mapSlice);
		zdzmap = dev_zdzmap;
		cx::write_raw(argv[6],zdzmap.data(),cryNum*cryDiffNum*detZdZNum);
	}

	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage: fullsim <mode> <threads> <blocks> <ngen> <seed> ....\n");
		printf(" mode = 1 <r1> <r2> generate smat files radius from <vr1>  to <vr2>-1\n");
		printf(" mode = 2 <outfile append or create> <r1> <r2> <ph1> <ph2> <z1> z2> <x> <y> <z> <dovol> <savemap> generate cylindrical phantom withe specified origin\n");
		return 0;
	}

	FILE * flog = fopen("fullsim.log","a");
	for(int k=0;k<argc;k++) fprintf(flog," %s",argv[k]); fprintf(flog,"\n"); fclose(flog);

	int argshift = 1;
	int mode = 1; mode= atoi(argv[1]);
	if(mode==2){
		do_phantom(argc,argv);
		return 0;
	}
	if(mode != 1) argshift = 0; // retro fit mode to old code 
	
	printf("Detector: len %.1f radius %.3f rings %d crytals/ring %d zstep %.3f phistep %.3f (degrees)\n",
		detLongLen,detRadius,zLongNum,cryNum,crySize,360.0/cryNum);

	uint threads = 256; if(argc>1) threads = atoi(argv[1+argshift]);
	uint blocks = 1024; if(argc>2) blocks  = atoi(argv[2+argshift]);

	uint size = blocks*threads;
	int passes = 1;
	long long  ngen = 1000000;
	int ndo = atoi(argv[3+argshift]);
	if(ndo <1000) ngen *= (long long)ndo;
	else {
		passes = ndo/1000;
		ngen *= 1000ll;
	}
	uint tries = (ngen+size-1)/size;
	ngen = (long long)tries*(long long)size;
	long long ngen_all = ngen*(long long)passes;
	std::random_device rd;
	long long seed = rd(); if (argc > 4) seed = atoi(argv[4+argshift]);

	int vx1 = 0;     if(argc > 5+argshift) vx1 = atoi(argv[5+argshift]);
	int vx2 = vx1+1; if(argc > 6+argshift) vx2 = atoi(argv[6+argshift]);

	Roi roi;
	roi.r.x =  voxSize*vx1;//roiRadius - voxSize; //0.0f;     //20*voxSize; //0.3f*roiRadius;
	roi.r.y =  roi.r.x+voxSize;           //roiRadius;           //voxSize;  //21*voxSize; //roiRadius;

	roi.z.x = crySize*(zNum-1);    // calib voxel spans z in range 63-64;
	roi.z.y =  roi.z.x+crySize;    // z span one detector ring;

	roi.phi.x = 0.0; 
	roi.phi.y = roi.phi.x+phiStep; 

	roi2xyz(roi);

	// use XORWOW
	thrustDvec<curandState> state(size);  // this for curand_states

	thrustHvec<uint>       hits(zLongNum*cryNum);
	thrustDvec<uint>   dev_hits(zLongNum*cryNum);
	thrustHvec<uint>       vfill(zLongNum*voxNum*voxNum);
	thrustDvec<uint>   dev_vfill(zLongNum*voxNum*voxNum);
	thrustHvec<ushort>     sfill(zLongNum*voxNum*voxNum);
	thrustDvec<ushort> dev_sfill(zLongNum*voxNum*voxNum);
	thrustHvec<uint>       map(mapSlice*mapSlice);
	thrustDvec<uint>   dev_map(mapSlice*mapSlice);
	thrustHvec<uint>       spot(spotNphi*spotNz*mapSlice);
	thrustDvec<uint>   dev_spot(spotNphi*spotNz*mapSlice);

	thrustDvec<double> dev_good(size);
	cx::MYTimer tim;
	init_generator<<<blocks, threads>>>(seed,state.data().get());

	for(int vx=vx1;vx<vx2;vx++){
		for(int k=0;k<passes;k++){
			//cylhits<<<blocks, threads >>>(dev_hits.data().get(), dev_vfill.data().get(), dev_good.data().get(), det, roi, state.data().get(), tries);
			cylhits<<<blocks, threads >>>(dev_hits.data().get(),dev_map.data().get(), dev_sfill.data().get(), dev_good.data().get(), roi, state.data().get(), tries);
		}
		checkCudaErrors( cudaPeekAtLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
		tim.add();
		find_spot<<<zNum,cryNum>>>(dev_map.data().get(),dev_spot.data().get(), roi);
		checkCudaErrors( cudaPeekAtLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );

		double all_good = thrust::reduce(dev_good.begin(), dev_good.end());
		
		double eff = 100.0*all_good/(double)ngen_all;
		printf("ngen %lld good %.0f eff %.3f%% time %.3f ms\n",ngen_all,all_good,eff,tim.time());

		hits = dev_hits;
		//vfill = dev_vfill;
		sfill = dev_sfill;
		//cx::write_raw("fsim_roi_int.raw",vfill.data(),zLongNum*voxNum*voxNum);
		//cx::write_raw("fsim_roi_short.raw",sfill.data(),zLongNum*voxNum*voxNum);
		//cx::write_raw("fsim_hits_int.raw",hits.data(),zLongNum*cryNum);

		//map = dev_map;
		//cx::write_raw("fsim_map_int.raw",map.data(),mapSlice*mapSlice);
		char name[256];
		sprintf(name,"spot_map%3.3d.raw",vx);
		spot = dev_spot;
		cx::write_raw(name,spot.data(),spotNphi*spotNz*mapSlice);
		if(vx < vx2-1){
			roi.r.x = roi.r.y;
			roi.r.y = roi.r.x+voxSize;
			clear_vec<<<blocks,threads>>>(dev_hits.data().get(),zLongNum*cryNum);
			clear_vec<<<blocks,threads>>>(dev_map.data().get(),mapSlice*mapSlice);
			clear_vec<<<blocks,threads>>>(dev_spot.data().get(),spotNphi*spotNz*mapSlice);
			clear_vec<<<blocks,threads>>>(dev_good.data().get(),size);
		}
	}

	printf("done\n");
	return 0;
}