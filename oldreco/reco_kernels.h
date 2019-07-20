#ifndef RECO_KERNELS_H_
#define RECO_KERNELS_H_

//-------  introduce some device global variables to save arguments --------------
//__device__ cudaSM *dev_sm;   // devglob.x count for all  attempts, globgen.y counts for for good events
__device__ __constant__ int dev_map8_x[8][3];
__device__ __constant__ int dev_map8_y[8][3];
__device__ __constant__ int dev_map8_c[8][2];

//--------------------------------------------------------------------------------

#define Dfix 1.0f

#include "smatrix.h"


struct cudaSM {
	SMfull_vox *v;
	smlor *lors;
	int voxels;
	int numlors;
};

// return rescaled value of lor[l] v[n]  // (no range check for speed)
__device__ float cuda_smval(cudaSM *sm,int vox,int l,int n)   { return (float)(sm->lors[sm->v[vox].lor_offset+l].v[n])*sm->v[vox].scale; }
__device__ uint  cuda_smkey(cudaSM *sm,int vox,int l)       { return sm->lors[sm->v[vox].lor_offset+l].key; }

__device__ void cuda_lor_from_key(uint key,hex &p)
{
	//allow 7-bits for z and 9-bits for c, total 32 bits
	p.c2 = key & 0x000001ff;
	key = key>>9;
	p.z2 = (key & 0x0000007f) -F120_SM_Zshift;
	key = key>>7;
	p.c1= key & 0x000001ff;
	key = key>>9;
	p.z1 = key - F120_SM_Zshift;
	return;
}

__device__ 	int cuda_hex_to_sector(int s,hex &p,hex &q)   //added 31/08/17
{
	q.x =  dev_map8_x[s][0]*p.x + dev_map8_x[s][1]*p.y + dev_map8_x[s][2];
	q.y =  dev_map8_y[s][0]*p.x + dev_map8_y[s][1]*p.y + dev_map8_y[s][2];
	q.c1 = dev_map8_c[s][0]+dev_map8_c[s][1]*p.c1;

	if (q.c1<0)q.c1 += 288;
	if (q.c1>=288)q.c1 -= 288;

	q.c2 = dev_map8_c[s][0]+dev_map8_c[s][1]*p.c2;
	if (q.c2<0)q.c2 += 288;
	if (q.c2>=288)q.c2 -= 288;

	q.z1 = p.z1;
	q.z2 = p.z2;
	if (q.z1==q.z2 && q.c1>q.c2){
		int t = q.c1;
		q.c1 = q.c2;
		q.c2 = t;
	}
	return 0;
}

__device__ int cuda_mirror(hex &m,int zbin)
{
	hex p = m;
	//m.x = p.x;  // mirror and proper lors share same parent voxel. 
	//m.y = p.y;
	m.z2 = zbin-p.z1;
	m.z1 = zbin-p.z2;
	if (m.z1 == m.z2){   // no c's swop in degenerate case to preserve proper state
		m.c1 = p.c1;
		m.c2 = p.c2;
	}
	else {             // but swop c's in non degenerate cases
		m.c2 = p.c1;
		m.c1 = p.c2;
	}
	//if (p.z1 == p.z2 && p.z1 == m.z1 && p.z1 == m.z2)               return 1; // mirror identical to origin (z1=z2=48  &xbin =96 only) Not an error
	//else if (m.z2 >= F120_NZ || m.z2 < 0 || m.z1 >= F120_NZ || m.z1 < 0) return 2; // mirror outside detector - possible off axis
	return 0;
}

__device__ int cuda_mirror(hex &m,hex &p,int zbin)
{
	m.x = p.x;
	m.y = p.y;
	m.z2 = zbin-p.z1;
	m.z1 = zbin-p.z2;
	if (m.z1 == m.z2){   // no c's swop in degenerate case to preserve proper state
		m.c1 = p.c1;
		m.c2 = p.c2;
	}
	else {             // but swop c's in non degenerate cases
		m.c2 = p.c1;
		m.c1 = p.c2;
	}
	return 0;
}


__device__ int cuda_make_dc(int c1,int c2)
{
	int dc = abs(c2-c1);
	if (c1 > c2) dc = F120_NXY-dc;  // fix logically negative dc values
	return dc-F120_DCmin;
}

__global__ void check_sm(cudaSM *sm)
{
	// commit suicide
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id==0){
		printf("voxels %d\n",sm->voxels);
		printf("numlors %d\n",sm->numlors);
		for (int k=0; k<11; k++){
			int offset = sm->v[k].lor_offset;
			uint key = cuda_smkey(sm,k,20);
			float v0 = cuda_smval(sm,k,20,0);
			float v1 = cuda_smval(sm,k,20,1);
			printf("v[%d] nx %d ny %d offset %d key %d val %f %f\n",k,sm->v[k].nx,sm->v[k].ny,offset,key,v0,v1);
		}
	}	
}

__global__ void lor_factors(float *meas,float *fp,int sets,int set)
{
	//int id = threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;

	// lors in [145][288][1176] order, one block processes [288][1176] items
	// subsets now defined by dc in 0-144 therfore skip blocks not in subset.
	// restore original osem at the sm lor level - do everything here
	

	int offset = blockIdx.x*F120_DZstride*F120_NXY;
	while (tid < F120_DZstride*F120_NXY){
		//if (blockIdx.x%sets != set) fp[offset+tid] = 0.0f;
		//else{
			float v = fp[offset+tid];
			if (v >0.0000000001f)fp[offset+tid] = meas[offset+tid]/v;
			else fp[offset+tid] = 0.0f;
		//}
		tid += blockDim.x;
	}
}

__global__ void vox_factors(float *teffs,float *act,float *bp,int sets,int set)
{
	// voxels szk format order [1161][95][8], kernel simply uses linear indexing
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	while (id <F120_SZKsize){
		float te = teffs[id];
		//float te = 100.0f;
		if (te > 0.000000001f) {
			float v = bp[id]*act[id]*1000000.0f/te;     //TODO fix teff=0
			act[id]= v;
		}
		id += blockDim.x*gridDim.x;
	}
}

__global__ void vox_factors_ref(float *teffs,float *act,float *bp,int sets,int set)
{
	int tid = threadIdx.x;
	// voxels int [128][128][95] array, one block process 128*95  items
	int offset = blockIdx.x*F120_NXYbins*F120_NZbins;
	int tvoffset = (sets+set-1)*F120_NXYbins*F120_NXYbins*F120_NZbins + offset;

	while (tid < F120_NXYbins*F120_NZbins){	
		float te = teffs[tvoffset+tid];
		//float te = 100.0f;
		if (te > 0.000000001f) {
			float v = bp[offset+tid]*act[offset+tid]*1000000.0f/te;     //TODO fix teff=0
			act[offset+tid]= v;
		}
		tid += blockDim.x;
	}
}

__global__ void backward_project_faster(cudaSM *sm,float *voxsum,float *zdzmap,int sets,int set,int kv)
{
	//__shared__ float div[2];
	//
	//  Evaluate the Numerator sum: Sum[v] =  SM[t][v] T[t]  summing over lors t for each voxel v
	//
	
	__shared__ float vshare[F120_NZbins*32];   // 32 = warp size

	int id = threadIdx.x + blockIdx.x*blockDim.x;

	// NO mirrors
	int idlor = id/16;             //  lor to process same for 16 threads per item
	int idsect = id%8;             //  which sector am I
	int even = (id%16)/8;          //  odd or even 0 or 1
	int idpart = threadIdx.x/16;   //  4 parts of blockDim = 64
	int idstep = blockDim.x*gridDim.x/16;
	int zstride = F120_NZbins*8;

	idlor = idlor*sets + set;   // for osem subset
	idstep *= sets;         // for osem subset

	hex p;
	hex q;

	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);

	int k = threadIdx.x;
	while (k< F120_NZbins*32){
		vshare[k]=0.0f;
		k += blockDim.x;
	}
	__syncthreads();

	int poffset = idpart*F120_NZbins*8;

	while (idlor < nlors){
		
		uint key = cuda_smkey(sm,kv,idlor);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idlor,even); // NB even =0 for odd(47) voxels and 1 for even(48) voxels 
		if (p.x==p.y) smval *= 0.5f;            
		//if (smval >0.0f){
			int dz = p.z2-p.z1;
			cuda_hex_to_sector(idsect,p,q);
			int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum
			int dcq = cuda_make_dc(q.c1,q.c2);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
			int sz_max = F120_TrueNZ-dz;

			for (int sz=0; sz<sz_max; sz++){    // zloop 
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				float tqval = zdzmap[dcq_offset+(sz+zsm_offset)];
				float cval = tqval*smval;

				//atomicAdd(&voxsum[kv*zstride+vzq*8+idsect],cval);

				vshare[poffset+vzq*8 + idsect] +=  cval;
				// no sync needed no intra warp collisions (even and odd voxels non overlap , buglet at ends?
			}
		//}
		__syncthreads();
		idlor += idstep;  // next lor
	}

	// reduce summed values
	//int chunks = 32/8;
	
	int j = threadIdx.x;
	while (j<zstride){
		vshare[j] += vshare[  zstride+j];
		vshare[j] += vshare[2*zstride+j];
		vshare[j] += vshare[3*zstride+j];
		j += blockDim.x;
	}
	__syncthreads();

	j = threadIdx.x;
	while (j <zstride){
		atomicAdd(&voxsum[kv*zstride+j],vshare[j]);
		j += blockDim.x;
	}

}

__global__ void backward_project_faster_ref(cudaSM *sm,float *voxsum,float *zdzmap,int sets,int set,int kv)
{
	//__shared__ float div[2];
	//
	//  Evaluate the Numerator sum: Sum[v] =  SM[t][v] T[t]  summing over lors t for each voxel v
	//
	// use 32 threads per lor each handles primary or mirror and even and odd

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	//int idt = id/32;   //  lor to process same for 32 threads per item
	//int ids = id%32;   //  which tread am i within 32 thread set: lor 8 sectors x 2 for proper & mirror x2 even/odd
	//int even = (id%32)/16; // 0 or 1
	//int idstep = blockDim.x*gridDim.x/32;

	// NO mirrors
	int idt = id/16;   //  lor to process same for 16 threads per item
	int ids = id%16;   //  which tread am i within 16 thread set: lor 8 sectors  x2 even/odd
	int even = (id%16)/8; // 0 or 1
	int idstep = blockDim.x*gridDim.x/16;

	idt = idt*sets + set;   // for osem subset
	idstep *= sets;         // for osem subset

	hex p;
	hex q;

	//if (threadIdx.x< 2 ) div[threadIdx.x] = 1.0f/sm->v[kv].geff[0];
	//__syncthreads();

	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);

	while (idt < nlors){
		//if (id == 0 && kv==75)printf("new idt %d\n",idt);
		uint key = cuda_smkey(sm,kv,idt);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idt,even);   // NB even =0 for odd(47) voxels and 1 for even(48) voxels // quite likely to work for full 32 thread warp
		if (p.x==p.y) smval *= 0.5f*Dfix;                // restored 02/11/17.
		//smval *= 0.000001f;
		//smval *= div[even];
		if (smval >0.0f){
			int dz = p.z2-p.z1;
			//int dc = p.c2-p.c1;
			int sector = (ids)%8;  // was(ids/2)%8
			cuda_hex_to_sector(sector,p,q);
			//if (ids%2) cuda_mirror(q,F120_TrueNZ-1+even);  // NO mirrors
			//if (dc%sets==set){
			int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum

			int dcq = cuda_make_dc(q.c1,q.c2);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
			//int dcq_offset = 10;
			//if (id <32 && kv==75) printf("%3d (%3d %3d %3d) %5d %5d\n",ids,q.c1,q.c2,dcq,dcq_offset,zsm_offset);
			int sz_max = F120_TrueNZ-dz;
			for (int sz=0; sz<sz_max; sz++){    // zloop (odd)
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				float tqval = zdzmap[dcq_offset+(sz+zsm_offset)];
				//float cval = tqval*smval/tvmap[vzq*F120_NXYstride+p.y*F120_NXYbins+p.x];
				float cval = tqval*smval;
				//if (p.x==p.y) cval *= 2.0f;  //30oct  pragmatic change  dont divide smval AND tqval by 2
				atomicAdd(&voxsum[vzq*F120_NXYstride+q.y*F120_NXYbins+q.x],cval);  // restored 02/11/17 

				//__syncthreads();  // Nothing shared beteen threads now? actually dont need this since all actors in same warp
			}
			//}
		}
		idt += idstep;  // next lor
	}
}


__global__ void backward_project_faster2(cudaSM *sm,float *voxsum,float *zdzmap,int sets,int set,int kv,int even)
{
	//__shared__ float div[2];
	//
	//  Evaluate the Numerator sum: Sum[v] =  SM[t][v] T[t]  summing over lors t for each voxel v
	//
	// use 32 threads per lor each handles primary or mirror and even and odd

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	//int idt = id/32;   //  lor to process same for 32 threads per item
	//int ids = id%32;   //  which tread am i within 32 thread set: lor 8 sectors x 2 for proper & mirror x2 even/odd
	//int even = (id%32)/16; // 0 or 1
	//int idstep = blockDim.x*gridDim.x/32;

	// NO mirrors
	int idt = id/8;   //  lor to process same for 16 threads per item
	int ids = id%8;   //  which tread am i within 16 thread set: lor 8 sectors  x2 even/odd
	//int even = (id%16)/8; // 0 or 1
	int idstep = blockDim.x*gridDim.x/8;

	idt = idt*sets + set;   // for osem subset
	idstep *= sets;         // for osem subset

	hex p;
	hex q;

	//if (threadIdx.x< 2 ) div[threadIdx.x] = 1.0f/sm->v[kv].geff[0];
	//__syncthreads();

	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);

	while (idt < nlors){
		//if (id == 0 && kv==75)printf("new idt %d\n",idt);
		uint key = cuda_smkey(sm,kv,idt);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idt,even);   // NB even =0 for odd(47) voxels and 1 for even(48) voxels // quite likely to work for full 32 thread warp
		if (p.x==p.y) smval *= 0.5f*Dfix;                // restored 02/11/17.
		//smval *= 0.000001f;
		//smval *= div[even];
		if (smval >0.0f){
			int dz = p.z2-p.z1;
			//int dc = p.c2-p.c1;
			int sector = (ids)%8;  // was(ids/2)%8
			cuda_hex_to_sector(sector,p,q);
			//if (ids%2) cuda_mirror(q,F120_TrueNZ-1+even);  // NO mirrors
			//if (dc%sets==set){
			int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum

			int dcq = cuda_make_dc(q.c1,q.c2);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
			//int dcq_offset = 10;
			//if (id <32 && kv==75) printf("%3d (%3d %3d %3d) %5d %5d\n",ids,q.c1,q.c2,dcq,dcq_offset,zsm_offset);
			int sz_max = F120_TrueNZ-dz;
			for (int sz=0; sz<sz_max; sz++){    // zloop (odd)
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				float tqval = zdzmap[dcq_offset+(sz+zsm_offset)];
				//float cval = tqval*smval/tvmap[vzq*F120_NXYstride+p.y*F120_NXYbins+p.x];
				float cval = tqval*smval;
				//if (p.x==p.y) cval *= 2.0f;  //30oct  pragmatic change  dont divide smval AND tqval by 2
				atomicAdd(&voxsum[vzq*F120_NXYstride+q.y*F120_NXYbins+q.x],cval);  // restored 02/11/17
				//atomicAdd(&voxsum[vzq*F120_NXYstride+p.y*F120_NXYbins+p.x+sector],cval); 
				//__syncthreads();  // Nothing shared beteen threads now? actually dont need this since all actors in same warp
			}
			//}
		}
		idt += idstep;  // next lor
	}
}



__global__ void backward_project(cudaSM *sm,float *voxsum,float *zdzmap,int kv,int even)
{
	//
	//  Evaluate the Numerator sum: Sum[v] =  SM[t][v] T[t]  summing over lors t for each voxel v
	//
	// use 16 threads per lor each handles primary or mirror

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int idt = id/16;   //  lor to process same for 8 threads per item
	int ids = id%16;   //  which tread am i within 8 thread set: lor 8 sectors x 2 for proper & mirror
	int idstep = blockDim.x*gridDim.x/16;

	hex p;
	hex q;

	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);

	while (idt < nlors){
		//if (id == 0 && kv==75)printf("new idt %d\n",idt);
		uint key = cuda_smkey(sm,kv,idt);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idt,even);   // NB even =0 for odd(47) voxels and 1 for even(48) voxels // quite likely to work for full 32 thread warp
		if (p.x==p.y) smval *= 0.5f*Dfix;                // fix for diagonal bug in sysmat  - present in all sectors thus counted twice in code below.
		smval *= 0.000001f;
		if (smval >0.0f){
			int dz = p.z2-p.z1;
			cuda_hex_to_sector(ids%8,p,q);
			if (ids>7) cuda_mirror(q,F120_TrueNZ-1+even);
			int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum

			int dcq = cuda_make_dc(q.c1,q.c2);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
			//if (id <16 && kv==75) printf("%3d (%3d %3d %3d) %5d %5d\n",ids,q.c1,q.c2,dcq,dcq_offset,zsm_offset);
			int sz_max = F120_TrueNZ-dz;
			for (int sz=0; sz<sz_max; sz++){    // zloop (odd)
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				float tqval = zdzmap[dcq_offset+(sz+zsm_offset)];
				atomicAdd(&voxsum[vzq*F120_NXYstride+q.y*F120_NXYbins+q.x], tqval*smval);
				//atomicAdd(&voxsum[(q.y*F120_NXYbins+q.x)*F120_NZbins+vzq], tqval*smval);  // worse on fermi card??
			}
			//__syncthreads();  // actually dont need this since all actors in same warp
		}
		idt += idstep;  // next lor
	}
}

template <int N> __global__ void forward_project_faster(cudaSM *sm,float *voxval,float *zdzmap,int sets,int set,int kv,int even)
{
	//
	//  Evaluate the denominator sum: Sum[t] =  SM[t][v] V[v]  summing over voxels v for each lor t
	//
	__shared__ float vshare[F120_TrueNZ+1][N];
	int *dcq_offshare =(int *)(&(vshare)[F120_TrueNZ][0]);  // reserve last (extra) slice

	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int idlor = id/8;        //  lor to process same for 16 threads per item
	int idsect = id%8;          //  which sector am i
	int idstep = blockDim.x*gridDim.x/8;
	int zstride = F120_NZbins*8;

	idlor = idlor*sets + set;   // for osem subset
	idstep *= sets;         // for osem subset

	hex p;
	hex q;
	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	//int vox_p_xypos =  p.y*F120_NXYbins+p.x;
	//if (ids==0 && kv == 0 )printf("idlor %d nlors %d\n",idlor,nlors);
	while (idlor < nlors){
		uint key = cuda_smkey(sm,kv,idlor);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idlor,even);   // NB even =0 for odd(47) voxels and 1 for even(48) voxels // quite likely to work for full 32 thread warp

		// fix for diagonal bug in sysmat  - present in all sectors thus counted twice in code below. 
		//if (p.x==p.y) smval *= 0.5f*Dfix;     // restored 02/11/17
		int dz = p.z2-p.z1;
		//int dc = p.c2-p.c1;
		int sz_max = F120_TrueNZ-dz;
		int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum

		if (smval >0.0f){

			cuda_hex_to_sector(idsect,p,q);

			//if (dc%sets==set){
			//int vox_xypos =  q.y*F120_NXYbins+q.x;   // added 12/10/17		
			int dcq = cuda_make_dc(q.c1,q.c2);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;

			dcq_offshare[threadIdx.x] = dcq_offset;
			for (int sz=0; sz<sz_max; sz++){    // zloop (odd)
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				//vshare[sz][threadIdx.x] = voxval[vzq*F120_NXYstride+vox_xypos]; 
				vshare[sz][threadIdx.x] = voxval[kv*zstride+vzq*8+idsect];  

			}

			//__syncthreads();  // all threads see this 02/11/17, actually dont need this since all actors in same warp

			int id_base = threadIdx.x & 0x00f8;
			for (int k=0; k<8; k++){     // loop over 8 sectors
				int dcq_offset = dcq_offshare[id_base+k];
				for (int j=0; j<sz_max; j+=8){   // was += 16
					int sz = j+idsect;
					if (sz<sz_max){
						//float qval = vshare[sz][id_base+k];
						atomicAdd(&zdzmap[dcq_offset+(sz+zsm_offset)],smval*vshare[sz][id_base+k]);  // restored 02/11/17
					}
				}
			}
			//}
		}
		//__syncthreads();  // all threads set this 02/11/17, actually dont need this since all actors in same warp

		idlor += idstep;  // next lor
	}
}

#if 0
__global__ void forward_project_even_faster(cudaSM *sm,float *voxval,float *zdzmap,int kv)
{
	//
	//  Evaluate the denominator sum: Sum[t] =  SM[t][v] V[v]  summing over voxels v for each lor t
	//
	__shared__ float vshare[F120_TrueNZ+1][64];
	int *dcq_offshare =(int *)(&(vshare)[F120_TrueNZ][0]);

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int idt = id/32;   //  lor to process same for 32 threads per item
	int ids = id%32;   //  which tread am i within 32 thread set: lor 8 sectors x 2 for proper & mirror
	int even = (id%32)/16; // 0 or 1
	int idstep = blockDim.x*gridDim.x/32;

	hex p;
	hex q;
	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	int vox_xypos =  p.y*F120_NXYbins+p.x;
	//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);
	while (idt < nlors){
		uint key = cuda_smkey(sm,kv,idt);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idt,even);   // NB even =0 for odd(47) voxels and 1 for even(48) voxels // quite likely to work for full 32 thread warp
		if (p.x==p.y) smval *= 0.5f;     // fix for diagonal bug in sysmat  - present in all sectors thus counted twice in code below.
		int dz = p.z2-p.z1;
		int sz_max = F120_TrueNZ-dz;
		int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum
		if (smval >0.0f){	
			cuda_hex_to_sector((ids/2)%8,p,q);
			if (ids%2) cuda_mirror(q,F120_TrueNZ-1+even);
			int dcq = cuda_make_dc(q.c1,q.c2);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
			dcq_offshare[threadIdx.x] = dcq_offset;
			for (int sz=0; sz<sz_max; sz++){    // zloop (odd)
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				vshare[sz][threadIdx.x] = smval*voxval[vzq*F120_NXYstride+vox_xypos];
				//atomicAdd(&zdzmap[dcq_offset+(sz+zsm_offset)],val*qval);  //17 or 15 with <<<64,64>>> secs  down to 8.9 when use atomicadd !?
			}
		}
		__syncthreads();  // actually dont need this since all actors in same warp
		int id_base = threadIdx.x & 0x00e0; // 0 or 32
		for (int k=0; k<32; k++){     // loop over 8 sectors for both proper and mirror lors
			int dcq_offset = dcq_offshare[id_base+k];
			for (int j=0; j<sz_max; j+=32){
				int sz = j+ids;
				if (sz<sz_max){
					//float qval = vshare[sz][id_base+k];
					atomicAdd(&zdzmap[dcq_offset+(sz+zsm_offset)],vshare[sz][id_base+k]);  // coherent access pattern!!  assumes val does not contain tube efficiencies
				}
			}
		}
		idt += idstep;  // next lor
	}
}
#endif



template <int N> __global__ void forward_project_faster_ref(cudaSM *sm,float *voxval,float *zdzmap,int sets,int set,int kv,int even)
{
//
//  Evaluate the denominator sum: Sum[t] =  SM[t][v] V[v]  summing over voxels v for each lor t
//
	__shared__ float vshare[F120_TrueNZ+1][N];
	int *dcq_offshare =(int *)(&(vshare)[F120_TrueNZ][0]);  // reserve last (extra) slice

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	//int idt = id/16;        //  lor to process same for 16 threads per item
	//int ids = id%16;        //  which tread am i within 16 thread set: lor 8 sectors x 2 for proper & mirror
	//int idstep = blockDim.x*gridDim.x/16;
	
	// no mirror now
	int idt = id/8;        //  lor to process same for 16 threads per item
	int ids = id%8;        //  which tread am i within 16 thread set: lor 8 sectors x 2 for proper & mirror
	int idstep = blockDim.x*gridDim.x/8;

	idt = idt*sets + set;   // for osem subset
	idstep *= sets;         // for osem subset

	hex p;
	hex q;
	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	//int vox_p_xypos =  p.y*F120_NXYbins+p.x;
	//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);
	while (idt < nlors){
		uint key = cuda_smkey(sm,kv,idt);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idt,even);   // NB even =0 for odd(47) voxels and 1 for even(48) voxels // quite likely to work for full 32 thread warp
	
		// fix for diagonal bug in sysmat  - present in all sectors thus counted twice in code below. 
		if (p.x==p.y) smval *= 0.5f*Dfix;     // restored 02/11/17
		int dz = p.z2-p.z1;
		//int dc = p.c2-p.c1;
		int sz_max = F120_TrueNZ-dz;
		int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum

		if (smval >0.0f){
			
			int sector = ids%8;
			cuda_hex_to_sector(sector,p,q);
			//if (ids>7) cuda_mirror(q,F120_TrueNZ-1+even);  // NO mirrors now
			
			//if (dc%sets==set){
			//int zsm_offset = 0;
			int vox_xypos =  q.y*F120_NXYbins+q.x;   // added 12/10/17		
			int dcq = cuda_make_dc(q.c1,q.c2);
			//if (dcq<0 || dcq>144) printf("bad dcq %d kv %d q (%2d %3d) (%2d %3d) p (%2d %3d) (%2d %3d) id %d idt %2d ids %2d smval %f\n",dcq,kv,q.z1,q.c1,q.z2,q.c2,p.z1,p.c1,p.z2,p.c2,id,idt,ids,smval);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
		
			dcq_offshare[threadIdx.x] = dcq_offset;
			for (int sz=0; sz<sz_max; sz++){    // zloop (odd)
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				//vshare[sz][threadIdx.x] = voxval[vzq*F120_NXYstride+vox_xypos]/tvmap[vzq*F120_NXYstride+vox_p_xypos];  // 12/10/17 BUG here vox_pos depends on q sector NOT p sector
				vshare[sz][threadIdx.x] = voxval[vzq*F120_NXYstride+vox_xypos];  // 12/10/17 BUG here vox_pos depends on q sector NOT p sector
				//if (p.x==p.y) vshare[sz][threadIdx.x] *= 0.5f*Dfix; //  restored 02/11/17; 

				//atomicAdd(&zdzmap[dcq_offset+(sz+zsm_offset)],val*qval);  //17 or 15 with <<<64,64>>> secs  down to 8.9 when use atomicadd !?
			}
		
			//__syncthreads();  // all threads see this 02/11/17, actually dont need this since all actors in same warp
		
			//int id_base = threadIdx.x & 0x00f0;  //just %16
			int id_base = threadIdx.x & 0x00f8;
			for (int k=0; k<8; k++){     // loop over 8 sectors for both proper and  but not mirrors mirror lors(was /16)
				int dcq_offset = dcq_offshare[id_base+k];
				for (int j=0; j<sz_max; j+=8){   // was += 16
					int sz = j+ids;
					if (sz<sz_max){
						//float qval = vshare[sz][id_base+k];
						atomicAdd(&zdzmap[dcq_offset+(sz+zsm_offset)],smval*vshare[sz][id_base+k]);  // restored 02/11/17
					}
				}
			}
			//}
		}
		//__syncthreads();  // all threads set this 02/11/17, actually dont need this since all actors in same warp

		idt += idstep;  // next lor
	}
}

__global__ void forward_project(cudaSM *sm,float *voxval,float *zdzmap,int kv,int even)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int idt = id/16;   //  lor to process same for 16 threads per item
	int ids = id%16;   //  which tread am i within 16 thread set: lor 8 sectors x 2 for proper & mirror
	int idstep = blockDim.x*gridDim.x/16;

	hex p;
	hex q;
	int nlors = sm->v[kv].lors;
	p.x =    sm->v[kv].nx;
	p.y =    sm->v[kv].ny;
	//int vox_xypos =  p.y*F120_NXYbins+p.x;
	//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);
	while (idt < nlors){
		uint key = cuda_smkey(sm,kv,idt);
		cuda_lor_from_key(key,p);
		float smval = cuda_smval(sm,kv,idt,even);   // NB even =0 for odd(47) voxels and 1 for even(48) voxels // quite likely to work for full 32 thread warp
		if (p.x==p.y) smval *= 0.5f*Dfix;     // fix for diagonal bug in sysmat  - present in all sectors thus counted twice in code below.
		
		if(smval >0.0f){                            
			int dz = p.z2-p.z1;
			cuda_hex_to_sector(ids%8,p,q);
			if (ids>7) cuda_mirror(q,F120_TrueNZ-1+even);
			int vox_xypos =  q.y*F120_NXYbins+q.x;

			//int zsm_offset = 0;
			
			int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum
			int dcq = cuda_make_dc(q.c1,q.c2);
			int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
			for (int sz=0; sz<F120_TrueNZ-dz; sz++){    // zloop (odd)
				int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
				vzq = min(F120_NZbins-1,max(0,vzq));
				float qval = voxval[vzq*F120_NXYstride+vox_xypos];			
				atomicAdd(&zdzmap[dcq_offset+(sz+zsm_offset)],smval*qval);  //17 or 15 with <<<64,64>>> secs  down to 8.9 when use atomicadd !?
			}
		}
		idt += idstep;  // next lor
	}
}

#endif