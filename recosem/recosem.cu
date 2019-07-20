// start very simple cylindrical geom
#pragma warning( disable : 4267)   // size_t int mismatch 
#pragma warning( disable : 4244)   // thrust::reduce int mismatch

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <stdlib.h>

#include "cx.h"
#include "timers.h"
#include "scanner.h"

//struct sm_part {
//	uint key;
//	float val;
//};

__host__ __device__  Lor  key2lor(uint key)
{
	Lor l;
	l.c2 = key & 0x000001ff;
	l.z2 = (key >> 9) & 0x0000007f;
	l.c1 = (key >> 16) & 0x000001ff;
	l.z1 = (key >> 25); // & 0x0000007f;  // Not necessary of unsigned shift?
	return l;
}

__host__ __device__ int cyc_diff(cint c1, cint c2) {
	return c1 > c2 ? c1-c2 : c2-c1;
}

 

// indexing modulo Range, assumes i is within interval
template <int Range> __host__ __device__ int cyc_sub(cint i, cint step) {
	return i >= step ? i - step : i - step + Range;
}

template <int Range> __host__ __device__ int cyc_dec(cint i) {
	return i >= 1 ? i - 1 : Range - 1;
}

template <int Range> __host__ __device__ int cyc_add(cint i, cint step) {
	return i + step < Range ? i + step : i + step - Range;
}

template <int Range> __host__ __device__ int cyc_inc(cint i) {
	return i + 1 < Range ? i + 1 : 0;
}

__host__ __device__ int c2_to_dc2(cint c1, cint c2) {
	//size_t cdc = cyc_sub<cryNum>(lor.c2,lor.c1) - cryDiffMin; // from phanom in fullsim
	return cyc_sub<cryNum>(c2,c1)-cryDiffMin;
}

__device__ int zdz_slice(int z1)
{
	return detZdZNum - (detShortRings-z1)*(detShortRings-z1+1)/2;
}

// assumes threads = cryNum i.e. 400 so that one thread blocks process all phis for fixed sm value
__global__  void forward_project(cr_Ptr<sm_part> sm, uint smstart, uint smend, cr_Ptr<uint> pet, cr_Ptr<float> vol,int ring, r_Ptr<float> K,int osteps,int ostep)
{
	int phi = threadIdx.x;
	uint smpos = smstart+blockIdx.x*osteps + ostep;

	while (smpos < smend) {
		Lor tl = key2lor(sm[smpos].key);
		tl.c1 = cyc_add<cryNum>(tl.c1, phi);     // rotate by phi		
		tl.c2 = cyc_add<cryNum>(tl.c2, phi);     // rotate by phi
		int dc = c2_to_dc2(tl.c1,tl.c2);          // Fix to sm	
		int tsum = tl.z1+tl.z2;
		float val= sm[smpos].val;
		
		for (int zs1 = 0; zs1 < detShortRings-tsum; zs1++) {  // zs1 is sliding posn of lh end of lor
			uint lor_index = (zdz_slice(zs1) + tsum)*cryCdCNum + dc*cryNum + tl.c1; // z2 not z
			if(pet[lor_index]>0){  // most lors zero for nice phantoms
				uint vol_index = (ring*detShortRings + zs1+tl.z1)*cryNum + phi;      // z+z1 here as voxel index
				float element = vol[vol_index] * val;  // need slice normalization factor here		
				atomicAdd(&K[lor_index],element);
			}
		}
		smpos += gridDim.x*osteps;
	}
}

// assumes threads = cryNum i.e. 400 so that one thread blocks process all phis for fixed sm value
__global__  void backward_project(cr_Ptr<sm_part> sm, uint smstart, uint smend, cr_Ptr<uint> pet, int ring, cr_Ptr<float> K, r_Ptr<float> M,int osteps,int ostep)
{
	int phi = threadIdx.x;
	uint smpos = smstart+blockIdx.x*osteps + ostep;  // offset ostep from start

	while (smpos < smend) {
		Lor tl = key2lor(sm[smpos].key);
		tl.c1 = cyc_add<cryNum>(tl.c1, phi);     // rotate by phi		
		tl.c2 = cyc_add<cryNum>(tl.c2, phi);     // rotate by phi
		int dc = c2_to_dc2(tl.c1,tl.c2);          // Fix to sm		
		int tsum = tl.z1+tl.z2;
		float val= sm[smpos].val;
		for (int zs1 = 0; zs1 < detShortRings-tsum; zs1++) {  // zs1 is sliding posn of lh end of lor
			uint lor_index = (zdz_slice(zs1) + tsum)*cryCdCNum + dc*cryNum + tl.c1; // z2 not z
			//uint lor_index = dc*cryNum*detZdZNum + (zdz_slice(tl.z1) + z)*cryNum  + tl.c1; // 10% faster
			if(pet[lor_index]>0){  // most lors zero for nice phantoms
				int vol_index = (ring*detShortRings + zs1+tl.z1)*cryNum + phi;       // z+z1 here as voxel index
				float element = val * pet[lor_index] / K[lor_index];  // val added 27/06/19!!
				atomicAdd(&M[vol_index],element);
			}
		}

		smpos += gridDim.x*osteps;  // skip osteps-1 lors
	}
}

__global__ void rescale(r_Ptr<float> vol, cr_Ptr<float> M, cr_Ptr<float> norm,int osteps,int ostep)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	//int nslice = ostep*radNum*detShortRings;
	while(id < detShortRings*radNum*cryNum){
		//vol[id] *= M[id] / norm[nslice + id/cryNum];
		vol[id] *= M[id] / (norm[id/cryNum]*osteps);
		id += blockDim.x*gridDim.x;
	}
}

template <typename T> __global__ void clear_vector(r_Ptr<float> a,uint len)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	while(id < len){
		a[id] = (T)0;
		id += blockDim.x*gridDim.x;
	}
}

int normalise_sm(thrustHvec<sm_part> &sm,thrustHvec<float> &norm,thrustHvec<uint> &smhits,thrustHvec<uint> &smstart,int osteps)
{
	// NB all dZ and dC cuuts assumed to have already been made in readspot
	uint norm_slice  = radNum*detShortRings;
	uint norm_size = norm_slice*osteps;
	// normalise allowing for voxel volume Router^2 = Rinner^2
	for(int r = 0;r<radNum;r++){
		uint sm_start = smstart[r];
		uint smnum = smhits[r+2];
		int ostep = 0;
		for(uint k=sm_start;k<sm_start+smnum;k++){
			Lor tl = key2lor(sm[k].key);
			float val = sm[k].val;
			for (int z = tl.z1; z < detShortRings - tl.z2; z++) {
				norm[norm_slice*ostep + r*detShortRings+z] += val;
			}	
			if(ostep < osteps-1) ostep++;
			else ostep = 0;
		}
	}
	printf("normalization done for %d rings and %d slices\n",radNum,detShortRings);	
	for(uint i=0;i<norm_size;i++) norm[i] /= 1.0e+10;  // assume 10^10 genrations per voxel
	cx::write_raw("norm_new.raw",norm.data(),norm_size);
	//cx::write_raw("norm_recip.raw",norm.data(),norm_size);
	return 0;
}

int list_sm(thrustHvec<sm_part> &sm,thrustHvec<uint> &smhits,thrustHvec<uint> &smstart)
{
	printf("list sm called\n");
	for(int r=0;r<radNum;r++){
		printf("list sm called r=%d\n",r);
		char name[256];
		sprintf(name,"smlist_r%3.3d.txt",r);
		FILE * flog = fopen(name,"w");
		uint sm_start = smstart[r];
		uint smnum = smhits[r+2];
		for(uint k=sm_start;k<sm_start+smnum;k++){
			Lor tl = key2lor(sm[k].key);
			float val = sm[k].val;
			fprintf(flog,"smpos %6u lor (%2d %3d)-(%2d %3d) val %.0f\n",k,tl.z1,tl.c1,tl.z2,tl.c2,val);
		}
		fclose(flog);
	}
	return 0;
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage reco <pet file (phantom)> <result file> <sm file> <sm nhits file> <iterations> [<osem steps|1>]\n");
		return 0;
	}

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//struct cudaFuncAttributes fa;
	//cudaError_t cudaResult = cudaSuccess;
	//cudaResult = cudaFuncGetAttributes(&fa,backward_project );
	//printf("cacheModeCA %u\n constSizeBytes %u\n localSizeBytes %u\n maxDynamicSharedSizeBytes %u\n 
	//              maxThreadsPerBlock %u\n numRegs %u\n preferredShmemCarveout %u\n",
	//		      fa.cacheModeCA,fa.constSizeBytes,fa.localSizeBytes,fa.maxDynamicSharedSizeBytes,fa.maxThreadsPerBlock,fa.numRegs,fa.preferredShmemCarveout);
	//return 0;

	int blscale = 1024; 
	int thscale = 256;  
	int niter = 10; if(argc> 5) niter = atoi(argv[5]);
	int osteps = 1; if(argc > 6) osteps = atoi(argv[6]); printf("OSEM steps = %d\n",osteps);
	// set up system matix
	char name[256];
	
	thrustHvec<uint> nhits(radNum+2);
	if(cx::read_raw(argv[4],nhits.data(),radNum+2)){printf("bad read %s\n",argv[4]); return 1;}	
	if(nhits[0] != radNum){printf("bad nhits = %d, expected %d\n",nhits[0],radNum); return 1;}
	thrustHvec<uint> nhits_start(radNum);
	nhits_start[0] = 0;
	for (int k = 1; k < radNum; k++) nhits_start[k] = nhits_start[k - 1] + nhits[1 + k];
	//for (int k = 0; k < radNum; k++) printf("hits start %3d %8u end %8u\n",k, nhits_start[k],nhits_start[k]+nhits[k+2]);
	// return 0;

	uint sm_size = nhits[1];
	uint lor_size = cryCdCNum*detZdZNum;
	uint vol_size = cryNum*radNum*detShortRings;
	uint norm_size = radNum*detShortRings*osteps;
	uint zphi_size = cryNum*detShortRings;
	printf("sm_size = %u, lor_size %u vol_size %u\n", sm_size,lor_size,vol_size);

	thrustHvec<sm_part>      sm(sm_size);
	thrustDvec<sm_part>  dev_sm(sm_size);
	if (cx::read_raw(argv[3], sm.data(), sm_size)) { printf("bad read on sm_file %s\n", argv[3]); return 1; }
	dev_sm = sm;

	thrustHvec<uint>      pet(lor_size);
	thrustDvec<uint>  dev_pet(lor_size);
	if (cx::read_raw(argv[1], pet.data(), lor_size)) { printf("bad read on pet file %s\n", argv[1]); return 1; }
	dev_pet = pet;

	thrustHvec<float>     K(lor_size); // working space for forward projection (voxels => lors)
	thrustDvec<float> dev_K(lor_size); 

	thrustHvec<float>     M(vol_size); // working space for backward projection  (lors => voxels)
	thrustDvec<float> dev_M(vol_size);

	thrustHvec<float>     vol(vol_size);
	thrustDvec<float> dev_vol(vol_size);

	thrustHvec<float>     norm(norm_size); // voxel normaliztions depend on ring and z
	thrustDvec<float> dev_norm(norm_size);

	cx::MYTimer ntim;
	ntim.start();
	normalise_sm(sm,norm,nhits,nhits_start,osteps);
	cx::write_raw("smnorm.raw",norm.data(),norm_size);
	dev_norm = norm;
	ntim.add();
	printf("Host normalize call %.3f ms\n",ntim.time());
	//list_sm(sm,nhits,nhits_start);
	//return 0;

	double tot_activity = 0.0;
	for (uint k = 0; k < lor_size; k++) tot_activity += pet[k];

	//float mean_activity = tot_activity / vol_size;
	//for (uint k = 0; k < vol_size; k++) vol[k] = mean_activity;

	// new initialisation accounting for voxel volumes (makes little difference)
	float roi_volume = cx::pi<float>*roiRadius*roiRadius;
	float act_density = tot_activity/roi_volume;
	//float act_pervox = tot_activity/vol_size;
	float r1 = 0.0f;
	float r2 = voxSize;
	for(int r=0;r<radNum;r++){
		float dr2 = r2*r2-r1*r1;
		float voxvol = cx::pi<float>*dr2/cryNum;
		for(uint k=0;k<zphi_size;k++) vol[r*zphi_size+k] = act_density*voxvol;
		//for(int k=0;k<zphi_size;k++) vol[r*zphi_size+k] = act_pervox;
		r1 = r2;
		r2 += voxSize;
	}

	dev_vol = vol;
	printf("total activity %.0f, activity density %.0f\n",tot_activity,act_density);
	//cx::write_raw("reco_start_vol.raw",vol.data(),vol_size);
	int threads = cryNum;
	int blocks = 512;
	
	cx::MYTimer tim1;
	cx::MYTimer tim2;
	cx::MYTimer tim3;
	cx::MYTimer all;

	all.reset();
	for(int iter = 0;iter< niter; iter++) for(int ostep = 0;ostep < osteps; ostep++){
		if(iter>0){
			clear_vector<float><<<blscale,thscale>>>(dev_K.data().get(),lor_size);
			clear_vector<float><<<blscale,thscale>>>(dev_M.data().get(),vol_size); 
		}
		tim1.reset();
		for (int r = 0; r < radNum; r++) {
			forward_project<<<blocks, threads>>>(dev_sm.data().get(), nhits_start[r], nhits_start[r]+nhits[r+2], dev_pet.data().get(), dev_vol.data().get(), r, dev_K.data().get(),osteps,ostep );
		}
		checkCudaErrors(cudaDeviceSynchronize());
		tim1.add();
		tim2.reset();
		for (int r = 0; r < radNum; r++) {
			backward_project<<<blocks, threads>>>(dev_sm.data().get(), nhits_start[r], nhits_start[r]+nhits[r+2], dev_pet.data().get(), r, dev_K.data().get(), dev_M.data().get(),osteps,ostep );
		}
		checkCudaErrors(cudaDeviceSynchronize());
		tim2.add();
		tim3.reset();
		rescale<<<blscale,thscale>>>(dev_vol.data().get(),  dev_M.data().get() , dev_norm.data().get(),osteps,ostep);
		checkCudaErrors(cudaDeviceSynchronize());
		tim3.add();
		//vol = dev_vol;
		//sprintf(name,"%s%3.3d.raw",argv[2],iter+1);
		//cx::write_raw(name, vol.data(), vol_size);
		checkCudaErrors(cudaDeviceSynchronize());
		all.add();
		printf("iteration %3d times %.3f %.3f %.3f all %.3f ms\n",iter+1,tim1.time(),tim2.time(),tim3.time(),all.time());
	}
	
	all.add();
	printf("All time %.3f ms\n", all.time());

	vol = dev_vol;
	sprintf(name,"%s_%d_final.raw",argv[2],niter);
	cx::write_raw(name, vol.data(), vol_size);

	//sprintf(name,"Kbug%3.3d.raw",niter);
	//K = dev_K;
	//cx::write_raw(name, K.data(), lor_size);

	//M = dev_M;
	//sprintf(name,"Mbug%3.3d.raw",niter);
	//cx::write_raw(name, M.data(), vol_size);

	return 0;
}
