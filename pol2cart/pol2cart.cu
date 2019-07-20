#pragma warning( disable : 4267)   // size_t int mismatch 
#pragma warning( disable : 4244)   // thrust::reduce int mismatch

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "curand_kernel.h"

#include "helper_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <random>

#include "cx.h"
#include "timers.h"
#include "scanner.h"

struct cp_grid {
	uint b[voxBox][voxBox];
	uint good;
	uint bad;
	int x; // carteisian origin
	int y;
	int phi;  // polar voxel
	int r;
};

struct cp_grid_map {
	float b[voxBox][voxBox];
	int x; // carteisian origin
	int y;
	int phi;  // polar voxel
	int r;
};

template <typename S> __global__ void init_generator(long long seed,S *states)
{
	// minimal version
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed + id, 0, 0, &states[id]);  
	//curand_init(seed, id , 0, &states[id]);  
}

template <typename S> __global__ void cpfill(float * cgrid, cp_grid * cp, int nxy, uint tries,S *states,int myp,int myr)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	cp_grid g;
	for(int i=0;i<voxBox;i++)for(int j=0;j<voxBox;j++)g.b[i][j] = 0;
	g.good = 0;
	g.bad = 0;
	g.phi = threadIdx.x;  // tid = phi
	g.r =   blockIdx.x;   // bid = r
	//g.phi = myp;
	//g.r =   myr;

	float p1 = (float)threadIdx.x*cx::pi2<float>/(float)cryNum;
	//float p1 = (float)myp*cx::pi2<float>/(float)cryNum;
	float p2 = p1 + cx::pi2<float>/(float)cryNum;
	float pmean = 0.5f*(p1+p2);
	float r1 = (float)blockIdx.x*voxSize;
	//float r1 = (float)myr*voxSize;
	float r2 = r1+voxSize;
	float rmean = 0.5f*(r1+r2);
	float x = rmean*cosf(pmean)+roiRadius;   //cos/sin swap 05/07/19
	float y = rmean*sinf(pmean)+roiRadius;
	int xc = (int)(x/voxSize+0.5f);
	int yc = (int)(y/voxSize+0.5f);
	g.x = max(0,min(xc-voxBoxOffset,nxy-voxBox));
	g.y = max(0,min(yc-voxBoxOffset,nxy-voxBox));

	//printf("myp/r %3d %3d p1/2 %.3f %.3f r1/2 %.1f %.1f x/y %.1f %.1f x/yc %3d %3d gx/y %3d %3d\n",myp,myr,p1,p2,r1,r2,x,y,xc,yc,g.x,g.y);

	float r1sq = r1*r1;
	float r2sq = r2*r2-r1sq;
	float dphi = cx::pi2<float>/(float)cryNum;
	S state = states[id];  // get state
	for(uint k=0;k<tries;k++){
		// generate decay point at a
		float phi = p1 + dphi*curand_uniform(&state);
		float r =   sqrtf(r1sq +r2sq*curand_uniform(&state));			
		float x = r*cosf(phi) + roiRadius;
		float y = r*sinf(phi) + roiRadius;
		//printf("generated x/y %.3f %.3f\n",x,y);
		if(x>=0 && x <= 2.0f*roiRadius && y>=0 && y <= 2.0f*roiRadius){
			int ix = (int)(x/voxSize)-g.x;
			int iy = (int)(y/voxSize)-g.y;
			if(ix<0 || ix >= voxBox || iy<0 || iy >= voxBox) {
				// printf("out of box error (%3d %3d) good %d p/r %.3f %.2f x/y %.2f %.2f ixy %d %d gxy %d %d\n",threadIdx.x,blockIdx.x,g.good,phi,r,x,y,ix,iy,g.x,g.y);
				g.bad++;
				continue;
			}
			else g.b[iy][ix]++;
			g.good++;
		}
		else {
			printf("out of roi error (%3d %3d) good %d p/r %.3f %.2f xy %.2f %.2f \n",threadIdx.x,blockIdx.x,g.good,phi,r,x,y);
			g.bad++;
			break;
		}
	}
	cp[id] = g;
	states[id] = state;  // save state so can continue
}

int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("usage pol2cart <pol2cart file> <seed> <nits> myp myr\n");
		return 0;
	}


	std::random_device rd;
	long long seed = rd(); if (argc > 2) seed = atoi(argv[2]);
	long long nits = 1000000; if(argc >3) nits *= atoll(argv[3]);
	int myp = 100; if(argc > 4) myp = atoi(argv[4]);
	int myr = 50;  if(argc > 5) myr = atoi(argv[5]);


	int nxy = (int)(voxNum + 0.5f);
	int csize = nxy*nxy;
	thrustHvec<float>          cgrid(csize);  // cartesian grid
	thrustDvec<float>      dev_cgrid(csize);

	int psize = cryNum*radNum;
	thrustHvec<cp_grid>          cpgrid(psize);  // mini grid around each r/phi voxel
	thrustDvec<cp_grid>      dev_cpgrid(psize);

	int threads = cryNum;
	int blocks =  radNum;
	printf("ready\n");
	// use XORWOW
	thrustDvec<curandState> states(psize);  // this for curand_states
	cx::MYTimer tim;
	init_generator<<<blocks,threads>>>(seed,states.data().get());
	cpfill<<<blocks,threads>>>(dev_cgrid.data().get(), dev_cpgrid.data().get(),nxy, nits,states.data().get(),myp,myr);
	checkCudaErrors(cudaDeviceSynchronize());
	tim.add();
	//cpfill<<<1,1>>>(dev_cgrid.data().get(), dev_cpgrid.data().get(),nxy, 10,states.data().get(),myp,myr);
	cpgrid =dev_cpgrid;
	//cx::write_raw(argv[1],cpgrid.data(),psize);
	//for(int k=0;k<psize;k++) cx::append_raw(argv[1],cpgrid[k].b,sizeof(uint)*9,0);
	int index = myr*cryNum+myp;
	printf("box[%3d][%3d] good %d bad %u\n",myr,myp,cpgrid[index].good,cpgrid[index].bad);
	printf("\n        "); for(int i=0;i<voxBox;i++) printf("   %3d   ",cpgrid[index].x+i); printf("\n");
	for(int i=0;i<voxBox;i++){
		printf(" %3d ",cpgrid[index].y+i);
		for(int j=0;j<voxBox;j++) printf(" %8u",cpgrid[index].b[i][j]);
		printf("\n");
	}
	printf("done time %.3f ms\n",tim.time());
	thrustHvec<cp_grid_map>          cpmap(psize);  // mini grid around each r/phi voxel
	for(int k=0;k<psize;k++){
		float div = 1.0f/(float)(cpgrid[k].good);
		cpmap[k].phi = cpgrid[k].phi;
		cpmap[k].r   = cpgrid[k].r;
		cpmap[k].x   = cpgrid[k].x;
		cpmap[k].y   = cpgrid[k].y;
		for(int i=0;i<voxBox;i++) for(int j=0;j<voxBox;j++) cpmap[k].b[i][j] = (float)cpgrid[k].b[i][j]*div;
	}
	printf("\n        "); for(int i=0;i<voxBox;i++) printf("   %3d   ",cpmap[index].x+i); printf("\n");
	for(int i=0;i<voxBox;i++){
		printf(" %3d ",cpmap[index].y+i);
		for(int j=0;j<voxBox;j++) printf(" %8.6f",cpmap[index].b[i][j]);
		printf("\n");
	}

	cx::write_raw(argv[1],cpmap.data(),psize);

	// for debug
	int vsize = voxBox*voxBox;
	thrustHvec<float> maps(psize*vsize);
	for(int k=0;k<psize;k++) for(int i=0;i<voxBox;i++) for(int j=0;j<voxBox;j++) maps[k*vsize+i*voxBox+j] =cpmap[k].b[i][j];
	cx::write_raw("cpdebug.raw",maps.data(),psize*vsize);

	return 0;
}