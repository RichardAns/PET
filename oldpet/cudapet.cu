
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h> 
#include <helper_math.h>
#include <device_functions.h>

#include "mystuff.h"
#include "aoptions.h"

//#include "F120_defs.h"
#include "F120_long_defs.h"

#include "cudapet.h"
#include "lors.h"
#include "zdzmap.h"
#include <algorithm>
#include <valarray>

#include <curand_kernel.h> 

#define offbug (6*23317)
// NB buffer now float4 thus SSlot 4 insead of 16 (or 6 instead of 24)

// use defines for kernal PET geometry
//#define F102_rmin 7.0

#include <Windows.h>
#include "cudapet_kernels.h"

int gen_with_cuda(double ngen,Source &s,AOptions &opt,Scanner &scn,FILE *logfile);
template <typename T> int bin_sino3d(T *s,T *ssrb,int c1,int z1,int c2,int z2,T hits,Scanner &scn);
double timePassed(LARGE_INTEGER &StartingTime);
int set_F120(Scanner &scn);
int simple_lor_file(AOptions &opt,uint *map,Scanner &scn,Source &s,int mfold,FILE *logfile,double trys,int compact,int cull);
int debug_sino(void);
double sum_map(uint *map,size_t &lors,int mfold);
int save_map_as_phantom(char *name,uint *amap,int mfold);
int comma_pad(double val, char *cpad);
int new_cull(int xpos,int ypos,int zpos,double cut,double tcut);
int append_existing(uint *amap, uint *dev_amap, Source &s, int mfoldchat, char *root);
void path_from_root(char *dest, char *root,int3 p);

//template<class T>  bool check_range(T value,T min,T max) {
//	return (value >= min) && (value <= max);
//}


int main(int argc, char *argv[])
{
	LARGE_INTEGER StartingTime;
	QueryPerformanceCounter(&StartingTime);
	
	Scanner  scn;
	set_F120(scn);

	//debug_sino();
	//return 0;

	if (argc < 2){
		printf("CudaPet %s - GPU event generation - options:  (NB units are mm)\n",F120_ID);
		printf("allsorts                      include storted files with compact (for debug)\n");
		printf("avol                          make3 D hist of decay points, 256x256x192 uint, default off\n");
		printf("avolbins nx ny nz             reset bins for decays histogram\n");
		printf("ahit                          make 3D hist of hit points, 1024x1024x95, default off\n");
		printf("amap                          make 3D lor map, (%dx%d)x(%dx%d) ushort, default off\n",F120_NXY,F120_NZ,F120_NXY,F120_NZ);
		printf("append                        append to existing map file if possible\n");
		printf("ccsave:n                      debug output for n passes, default none\n");
		printf("compact                       minimum ouput for production\n");
		printf("cyl                           active vol cylinder\n");
		printf("cylrad r                      cylinder radius, default 41.0\n");
		printf("cylpos x y z                  cylinder centre, default 0.0 0.0 zmax/2 \n");
		printf("cyllen l                      cylinder length along z axis, default zmax\n");

		printf("docry                         output debug docry file\n");
		printf("doi                           use interation points not entry faces\n");
		printf("EdgeTol:value                 relex inside block test to include faces, default 0.001\n");
		printf("ggen:value                    approx events time 10^9 overides npass, default 1.0\n");
		printf("genrot:val                    rotate generated lors by val degrees wrt crystals (default 7.5)\n");
		printf("highres                       voxel x-y dims *0.5, z dim unchanged\n");
		
		printf("lors                          make lorfile from amap\n");     
		printf("minigen                       short gensys info (implied by compact)\n");
		printf("mfold:nz                      reduce amapsize by folding at nz, assumes small voxel with z1 <=nz+1 & z2 >= nf and z2>=z1 \n");
		printf("mapsum                        single slice summed amap (%dx%d) ushort,  default off\n",F120_NXY,F120_NZ);

		printf("ngen:value                    number of passes for, default 20 small runs\n");
		printf("phantom                       this run to make phantom dataset (forces ranseed)\n");
		printf("ranseed                       random seed using tickcount\n");
		printf("reset                         cause timeout - resets card???\n");
		printf("rootpath <path>               root for compact generation, e.g D:\\temp\\nextgen\n");

		printf("setseed:value                 set seed (override ranseed if present or implied)\n");
		printf("sphere                        active vol sphere\n");
		printf("sphrad r                      sphere radius, default 1.0\n");
		printf("sphpos x y z                  sphere centre, default 0.0 0.0 zmax/2\n");
		printf("sinos                         make F120 3D sinograms imples amap\n");

		printf("vox                           active vol cuboid voxel\n");
		printf("voxpos x y z                  vox centre, default 0.0 0.0 zmax/2 \n");
		printf("voxnum nx ny nz               vox number, valid nx and ny are in [0,%d], nz is %d or %d\n",F120_NXYbins-1,(F120_NZlongbins-1)/2,(F120_NZlongbins+1)/2);
		printf("voxdim xd yd zd               voxel dimensions, default %.3f %.3f %.3f\n",F120_XYBin,F120_XYBin,F120_ZBin);
		
		printf("voxtube xl yl xh yh           voxel tube (for testing) eg 64 64 66 70 for min and max corners\n");
	   
		return 0;
	}

	char cudalog_name[] = "D:\\logfiles\\cudapet.log";
	FILE *logfile = fopen(cudalog_name,"a");
	if (!logfile) {
		logfile = fopen(cudalog_name,"w");
		if (logfile) printf("new %s logfile created\n",cudalog_name);
	}
	if (!logfile) { printf("can't open %s",cudalog_name);  return 1; }
	fprintf(logfile,"cudapet %s version 2.1 (1/2 block rotn) args: ",F120_ID); 
	for (int k=0; k<argc; k++) fprintf(logfile," %s",argv[k]); 
	fprintf(logfile,"\n");

	AOptions opt(argc,argv,1);
	

	if (opt.isset("verbose"))opt.set_verbose(1);
	if (opt.isset("reset")){
		printf("provoke time out to reset card!\n");
		fclose(logfile);
		time_out<<<16,128>>>();
		return 0;
	}
	//float tol = 0.001f;

	double ngen = (1000000000/(NThreads*NBlocks*MSlot));
	if (opt.isset("ngen") && atof(opt.get_string("ngen")) > 0.0)  ngen = atof(opt.get_string("ngen"));
	else if (opt.isset("ggen") && atof(opt.get_string("ggen")) > 0.0) ngen = ngen*atof(opt.get_string("ggen"));

	Source s(opt,scn);

	gen_with_cuda(ngen,s,opt,scn,logfile);

	printf("Total time taken %.3f secs\n",timePassed(StartingTime));
	fprintf(logfile,"\n");
	fclose(logfile);
	return 0;
}

// this for both device and host. Note buffers are set to zeros
template <typename T> int make_buffers(T **buf_out,T **dev_buf_out, size_t len, char *tag)
{
	T *buf = (T *)calloc(len,sizeof(T));
	if (!buf) { printf("calloc error %s\n",tag); return 1; }
	T *dev_buf = NULL;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_buf,len*sizeof(T));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc dev_%s failed [%s]\n",tag,cudaGetErrorString(cudaStatus)); return 1; }
	cudaStatus = cudaMemset(dev_buf,0,len*sizeof(T));
	if (cudaStatus != cudaSuccess) { printf("cudaMemset to dev_%s failed [%s]",tag,cudaGetErrorString(cudaStatus));  return 1; }

	// hairy pointer syntax thanks to cuda
	*buf_out = buf;
	*dev_buf_out = dev_buf;
	return 0;
}

// this for both device and host. Note buffers are set to zeros
template <typename T> int make_buffers_update(char *name,T **buf_out,T **dev_buf_out,size_t len)
{
	T *buf = (T *)malloc(len*sizeof(T));
	if (!buf) { printf("malloc for %s failed\n",name); return 1; }
	read_raw_or_clear<T>(name,buf,len);

	T *dev_buf = NULL;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_buf,len*sizeof(T));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc for %s failed [%s]\n",name,cudaGetErrorString(cudaStatus)); return 1; }

	cudaStatus = cudaMemcpy(dev_buf,buf,len*sizeof(T),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy for %s failed [%s]",name,cudaGetErrorString(cudaStatus));  return 1; }

	// hairy pointer syntax thanks to cuda
	*buf_out = buf;
	*dev_buf_out = dev_buf;
	return 0;
}


template <typename T> int make_buffer(T **buf_out, size_t len, char *tag)
{
	T *buf = (T *)calloc(len,sizeof(T));
	if (!buf) { printf("calloc error %s\n",tag); return 1; }

	// hairy pointer syntax
	*buf_out = buf;
	return 0;
}

// NB Source is passed by value here
int gen_with_cuda(double ngen,Source &s,AOptions &opt,Scanner &scn,FILE *logfile)
{
	char name[256];

	float tol =     opt.set_from_opt("edgetol",0.001f);
	int MFold_XY = F120_STride;
	int MFold_Z =  F120_STride;
	int mfold =   opt.set_from_opt("mfold",0);
	if (mfold > 0 && mfold < F120_NZ){
		MFold_XY = (F120_NZ-mfold)*F120_NXY;  //z2 c2 range    mfold <= z2 < NZ and z2 >= z1 (wlg by lor end sorting)
		MFold_Z = (mfold+2)*F120_NXY;         // z1 c1 range   0 <= z1 <= mfold+1
		printf("Using mfold = %d folded amap of dimensions %dx%d\n",mfold,MFold_XY,MFold_Z);
		if (mfold != 47) printf("WARNING mfold != 47 this might not work...\n");
	}
	else mfold = 0;

	int3 avsize ={ 256,256,256 };   // default size

	int dophantom = opt.isset("phantom") ? 1 : 0;
	int dovol =     opt.isset("avol") ? 1 : 0;
	int dohit =     opt.isset("ahit") ? 1 : 0;
	int dosino =    opt.isset("sinos")   ? 1 : 0;
	int dolors =    opt.isset("lors")    ? 1 : 0;
	int domap =     opt.isset("amap")    ? 1 : 0;
	int domapsum =  opt.isset("mapsum")  ? 1 : 0;
	int dodoi =     opt.isset("doi")     ? 1 : 0;
	int docry =     opt.isset("docry")   ? 1 : 0;
	int compact =   opt.isset("compact") ? 1 : 0;
	if (dosino || dolors || domapsum ) domap = 1;  // will check again before outputting amap
	if (dophantom) {
		dovol = 1;
		domap = 0;  // this is now incompatable with phantom
		domapsum = 0;
		avsize ={ 128,128,F120_NZ*4 };   // use 1/4 crystal bins to capture end effects in short 1/2 bin ROI
	}
	else if (opt.isset("avolbins")){
		int k = opt.isset("avolbins");
		avsize.x = opt.iopt(k+1);
		avsize.y = opt.iopt(k+2);
		avsize.z = opt.iopt(k+3);
	}

	//int smap_save = opt.set_from_opt("smap",1000);
	int ccsave =    opt.set_from_opt("ccsave",-1);

	char root[256];
	if (opt.isset("rootpath")) strcpy(root,opt.get_string("rootpath"));
	else strcpy(root,"default");
	printf("using root = %s\n",root);

	printf("%s using tol %.6f, ngen %.0f, avol %d, ahit %d, amap %d, cry %d, doi %d, ccsave %d fold %d mapsum %d\n",F120_ID,tol,ngen,dovol,dohit,domap,docry,dodoi,ccsave,mfold,domapsum);
	printf("source is %d pos: %.3f %.3f %.3f, dim %.3f %.3f %.3f\n",s.t,s.o.x,s.o.y,s.o.z,s.dim.x,s.dim.y,s.dim.z);
	
	cudaError_t cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr,"cudaSetDevice failed [%s]! ",cudaGetErrorString(cudaStatus));
		return 1;
	}
	//printf("got device\n");
	// these for results
	uint *avol = NULL;
	uint *dev_avol = NULL;
	float xlow = -F120_XYBin*(float)(avsize.x/2);
	float xhigh = F120_XYBin*(float)(avsize.x/2);
	float zlow = 0.0f;
	float zhigh = F120_BZface;
	float4 limits = make_float4(xlow,xhigh,zlow,zhigh);
	if (dovol) printf("plotting volume is %d x %d x %d avol limits x/y: %.3f %.3f z: %.3f %.3f\n",avsize.x,avsize.y,avsize.z,xlow,xhigh,zlow,zhigh);
	//float4 limits = make_float4(-F120_XYBin*2.0f,F120_XYBin*6.0f,F120_Csize*45.0f,F120_Csize*49.0f);
	if (dophantom) { if (make_buffers_update<uint>("phantvol.raw",&avol,&dev_avol,avsize.x*avsize.y*avsize.z)) return 1; }
	else if (dovol){ if (make_buffers<uint>(&avol,&dev_avol,avsize.x*avsize.y*avsize.z,"avol"))                return 1; }
	
	uint *ahit = NULL;
	uint *dev_ahit = NULL;
	float4 limits_hits = make_float4(-85.0f,85.0f,0.0f,F120_BZface);
	if (dohit) if(make_buffers<uint>(&ahit,&dev_ahit,1024*1024*95,"ahit")) return 1;	

	uint *amap = NULL;
	uint *dev_amap = NULL;
	//int ismap = 0;
	if (dophantom)           { if(make_buffers_update<uint>("phantom.raw",&amap,&dev_amap,F120_DCsize*F120_NXY*F120_DZstride)) return 1; } 
	else if (domap && mfold) { if(make_buffers<uint>(&amap,&dev_amap,MFold_XY*MFold_Z,"folded amap")) return 1; }      // uint folded amap
	else if (domap)          { if(make_buffers<uint>(&amap,&dev_amap,F120_STride*F120_STride,"amap")) return 1; }      // uint amap

	if (s.t == VOXEL && opt.isset("append")) append_existing(amap,dev_amap,s,mfold,root);

	uint *mapsum = NULL;
	if (domapsum) if (make_buffer<uint>(&mapsum,F120_STride*3,"mapsum")) return 1;  // full size even if folded

	float4 *bigbuf = NULL;
	float4 *dev_bigbuf = NULL;
	if(make_buffers<float4>(&bigbuf,&dev_bigbuf,BSize,"bigbuf")) return 1;

	uint4 *gentrys    =  NULL;
	uint4 *dev_gentrys = NULL;
	if (make_buffers<uint4>(&gentrys,&dev_gentrys,NThreads*NBlocks,"gentrys")) return 1;
	cudaStatus = cudaMemcpyToSymbol(devglob_gentrys, &dev_gentrys, sizeof(dev_gentrys));
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol dgentrys failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	// Note very elegent copy of stucture to __device__ __constant__ memory. 
	// Unfortunately performance 21% slower before using macros. 
	// Therefore kernels use #defined macros for now.
	cudaStatus = cudaMemcpyToSymbol(dev_scn,&(scn),sizeof(Scanner));
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol scn failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	float2 *dbuf = NULL;     // these are only needed if dodoi is set
	float2 *dev_dbuf = NULL;
	if(dodoi) if (make_buffers<float2>(&dbuf,&dev_dbuf,DSize,"dbuf")) return 1;

	int4 *crybuf = NULL;   // these are only needed if docry set
	int4 *dev_crybuf = NULL;
	if(docry) if (make_buffers<int4>(&crybuf,&dev_crybuf,CSize,"crybuf")) return 1;

	// this defines one block in standard position
	// tol uses widen block to allow for rounding errors when ray is at a face
	// NB block centre is North (x=0, y=Rmin), front face at z=0 back face z= BZface
	float *corners = (float *)malloc(12*sizeof(float));  // two copies exact and relaxed
	if (!corners) { printf("corners malloc error\n"); return 1; }
	corners[0] = -0.5f*F120_BPface;       corners[ 6] = corners[0] - tol;
	corners[1] =  0.5f*F120_BPface;       corners[ 7] = corners[1] + tol;
	corners[2] =  F120_Rmin;              corners[ 8] = corners[2] - tol;
	corners[3] =  F120_Rmin+F120_Rsize;   corners[ 9] = corners[3] + tol;
	corners[4] =  0.0f;                   corners[10] = corners[4] - tol;
	corners[5] =  F120_BZface;            corners[11] = corners[5] + tol;

	float *dev_corners = NULL;   // load corners to device array
	cudaStatus = cudaMalloc((void**)&dev_corners,12*sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc dev_corners failed [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	cudaStatus = cudaMemcpy(dev_corners,corners,12*sizeof(float),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to corners failed [%s]",cudaGetErrorString(cudaStatus));  return 1;	}

	curandState *devStates;  // load random number states to device array
	cudaStatus = cudaMalloc((void **)&devStates,NBlocks*NThreads*sizeof(curandState));
	long long seed = opt.set_from_opt("setseed",12345);
	//if ( (opt.isset("setseed") == 0) && (opt.isset("ranseed") || opt.isset("phantom")) ) seed = GetTickCount();
	if (opt.isset("ranseed")) seed = GetTickCount();
 
	setup_randstates<<<NBlocks,NThreads>>>(devStates,seed);	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"setup_randstates launch failed: [%s]\n",cudaGetErrorString(cudaStatus));	return 1;}

	//printf("after corners\n");

	FILE* cc = NULL;  FILE *ccy = NULL;  FILE *ccd = NULL;
	if(ccsave >0) {
		cc = fopen("cc.raw","wb");
		if(docry) ccy = fopen("ccy.raw","wb");
		if(dodoi) ccd = fopen("ccd.raw","wb");
	}
	LARGE_INTEGER Loop_StartingTime;
	QueryPerformanceCounter(&Loop_StartingTime);

	//printf("at gen loop start\r");

	// this is main generate loop
	int nc=1;
	for (double z=0; z<ngen; z+=1.0){     // default ngen 8139 ~ 10^9 events
		if (nc%200==0){ printf("in genloop at z= %.0f\r",z); nc=1; }
		else nc++;
		//  provide defaults if rays miss
		//cudaStatus = cudaMemset(dev_bigbuf,0,BSize*sizeof(float4));
		fill_bigbuf<<<NBlocks,NThreads>>>(dev_bigbuf,-2.0f);
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemset bigbuf failed [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z);  return 1; }
	
		// generate back-to-back gamma pair and swim to rmin cyinder
		if (dophantom) generate_kernel2_phantom<<<NBlocks,NThreads>>>(devStates,dev_bigbuf,s,dev_avol,limits,avsize.x,avsize.z);
		else generate_kernel2<<<NBlocks,NThreads>>>(devStates,dev_bigbuf,s);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_kernel2 launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }

		// find path in block1 and possible continuation into second block (about 25%)
		int blocks = NBlocks*NThreads*MSlot/32;
		generate_kernel3<<<blocks,256>>>(dev_bigbuf,dev_corners);  // using 8 threads per item
		//generate_kernel3<<<blocks,192>>>(dev_bigbuf,dev_corners);    // using 6  threads per item
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_kernel3 launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }

		
		if (docry){
			// get initial crystal entry and exit positions 
			generate_kernel4<<<16,NThreads>>>(dev_bigbuf,dev_crybuf,dev_corners);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_kernel4 launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }
		}

		if (dodoi){
			// get crystal interations
			generate_kernel5<<<16,NThreads>>>(devStates,dev_bigbuf,dev_dbuf);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_kernel5 launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }
		}


		// histogram results
		if (dovol && !dophantom){
			int hblocks = NBlocks*NThreads*MSlot/256;
			generate_avol<<<hblocks,256>>>(dev_bigbuf,dev_avol,limits,avsize.x,avsize.z);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_avol launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }
		}

		if (dohit){
			int hblocks = NBlocks*NThreads*MSlot/256;
			generate_ahits<<<hblocks,256>>>(dev_bigbuf,dev_ahit,limits_hits,1024,95);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_avol launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }
		}
		if (dophantom){
			generate_zdzmap<<<16,NThreads>>>(dev_bigbuf,dev_amap,dev_corners);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_zdzmap launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }
		}
		if (domap){
			if (dodoi){
				if (mfold == 0){ generate_amap_doi<<<16,NThreads>>>(dev_bigbuf,dev_dbuf,dev_amap,dev_corners); }
				else           { generate_fmap_doi<<<16,NThreads>>>(dev_bigbuf,dev_dbuf,dev_amap,dev_corners,mfold); }
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_amap_doi  launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }
			}
			else{
				if (mfold==0){ generate_amap<<<16,NThreads>>>(dev_bigbuf,dev_amap,dev_corners); }
				else         { generate_fmap<<<16,NThreads>>>(dev_bigbuf,dev_amap,dev_corners,mfold); }
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) { fprintf(stderr,"generate_amap launch failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z); return 1; }
			}
		}

		// this for debug
		if (z< ccsave){
			cudaStatus = cudaMemcpy(bigbuf,dev_bigbuf,BSize*sizeof(float4),cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to bigbuf failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z);	return 1; }
			fwrite(bigbuf,sizeof(float4),BSize,cc);
			if (docry){
				cudaStatus = cudaMemcpy(crybuf,dev_crybuf,CSize*sizeof(int4),cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to crybuf failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z);	return 1; }
				fwrite(crybuf,sizeof(int4),CSize,ccy);
			}
	
			if (dodoi){
				cudaStatus = cudaMemcpy(dbuf,dev_dbuf,DSize*sizeof(float2),cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to dbuf failed: [%s] for z=%d\n",cudaGetErrorString(cudaStatus),z);	return 1; }
				fwrite(dbuf,sizeof(float2),DSize,ccd);
			}
		}
	}  //end event generation loop
	printf("end genloop at z= %.0f\n",ngen);

	cudaDeviceSynchronize();  // necessary for timing 
	printf("generate time %.3f secs, seed %ld\n",timePassed(Loop_StartingTime),seed);
	fprintf(logfile,"generate time %.3f secs, seed %ld,",timePassed(Loop_StartingTime),seed);
	if (cc) { fclose(cc); printf("cc.raw written\n"); }
	if (ccy) { fclose(ccy); printf("ccy.raw written\n"); }
	if (ccd) { fclose(ccd); printf("ccd.raw written\n"); }
	LARGE_INTEGER IO_StartingTime;
	QueryPerformanceCounter(&IO_StartingTime);

	if (dovol){
		cudaStatus = cudaMemcpy(avol,dev_avol,avsize.x*avsize.y*avsize.z*sizeof(uint),cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to avol failed: %s\n",cudaGetErrorString(cudaStatus));	return cudaStatus; }
		if(dophantom)write_raw<uint>("phantvol.raw",avol,avsize.x*avsize.y*avsize.z);
		else         write_raw<uint>("avol.raw",avol,avsize.x*avsize.y*avsize.z);
		//FILE *ah = fopen("avol.raw","wb");
		//fwrite(avol,sizeof(uint),256*256*256,ah);
		//fclose(ah);
		//printf("avol.raw written\n");
		free(avol);
		cudaFree(dev_avol);
	}

	if (dohit){
		cudaStatus = cudaMemcpy(ahit,dev_ahit,1024*1024*95*sizeof(uint),cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to ahit failed!");	return cudaStatus; }
		printf("memcopy ahit done\n");
		FILE *ah = fopen("ahit.raw","wb");
		fwrite(ahit,sizeof(uint),1024*1024*95,ah);
		fclose(ah);
		printf("ahit.raw written\n");
		free(ahit);
		cudaFree(dev_ahit);
	}

	if (dophantom){
		size_t zdzmap_size = F120_DCsize*F120_NXY*F120_DZstride;
		cudaStatus = cudaMemcpy(amap,dev_amap,zdzmap_size*sizeof(uint),cudaMemcpyDeviceToHost);  
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to zdzmap failed! [%s] at end",cudaGetErrorString(cudaStatus)); return 1; }
		write_raw<uint>("phantom.raw",amap,zdzmap_size);
	}

	// always get statistics now
	double trys[5] ={ 0.0,0.0,0.0,0.0,0.0 };
	cudaStatus = cudaMemcpy(gentrys,dev_gentrys,NThreads*NBlocks*sizeof(uint4),cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to gentrys failed! [%s] at end",cudaGetErrorString(cudaStatus)); return 1; }
	//FILE *gt = fopen("gentrys.raw","wb");
	uint *t = (uint *)malloc(NThreads*NBlocks*sizeof(uint4));  // NB reshaping arrays here
	for (int k = 0; k<NThreads*NBlocks; k++) {
		t[k] = gentrys[k].x;
		t[k+NThreads*NBlocks] = gentrys[k].y;
		t[k+2*NThreads*NBlocks] = gentrys[k].z;
		t[k+3*NThreads*NBlocks] = gentrys[k].w;
		trys[0] += gentrys[k].x;
		trys[1] += gentrys[k].y;
		trys[2] += gentrys[k].z;
	}

	if(opt.isset("gentrys")) write_raw<uint>("gentrys.raw",t,NThreads*NBlocks*4);
	free(t);
	
	if (domap){

		if (mfold==0) cudaStatus = cudaMemcpy(amap,dev_amap,F120_STride*F120_STride*sizeof(uint),cudaMemcpyDeviceToHost);       // uint
		else          cudaStatus = cudaMemcpy(amap,dev_amap,MFold_XY*MFold_Z*sizeof(uint),cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to amap failed! [%s] at end",cudaGetErrorString(cudaStatus)); return 1; }
		if (dosino){  // write sinos using full amap
			uint *sino3d = (uint *)calloc(F120_SNX*F120_SNY*F120_SNZ,sizeof(uint));
			if (!sino3d) { printf("calloc error sino3d\n"); return 1; }
			uint *ssrb = (uint *)calloc(F120_SNX*F120_SNY*F120_SNZ2,sizeof(uint));
			if (!ssrb) { printf("calloc error ssrd\n"); return 1; }
			int k = 0;
			if (mfold==0){
				for (int z1=0; z1<F120_NZ; z1++) for (int c1=0; c1<F120_NXY; c1++) for (int z2=0; z2<F120_NZ; z2++) for (int c2=0; c2<F120_NXY; c2++) {
					uint hits = amap[k++];
					if (hits > 0) bin_sino3d(sino3d,ssrb,c1,z1,c2,z2,hits,scn);
				}
			}
			else {
				for (int z1=0; z1<mfold+2; z1++) for (int c1=0; c1<F120_NXY; c1++) for (int z2=mfold; z2<F120_NZ; z2++) for (int c2=0; c2<F120_NXY; c2++) {
					uint hits = amap[k++];
					if (hits > 0) bin_sino3d(sino3d,ssrb,c1,z1,c2,z2,hits,scn);
				}

			}
			//FILE *sss = fopen("sino3d.raw", "wb");
			//fwrite(sino3d, sizeof(uint),F120_SNX*F120_SNY*F120_SNZ,sss);
			//fclose(sss); printf("sino3d old write\n");
			write_raw<uint>("sino3d.raw",sino3d,F120_SNX*F120_SNY*F120_SNZ);
			write_raw<uint>("ssrb.raw",ssrb,F120_SNX*F120_SNY*F120_SNZ2);
			free(sino3d);
			free(ssrb);
		}
		int docull = 0;
		if (opt.isset("cull")) docull = 1;
		if (opt.isset("lors"))  simple_lor_file(opt,amap,scn,s,mfold,logfile,trys[2],compact,docull);  // trys not set here

		if (opt.isset("amap")){    //  dont write this big file if it is only implicly created for sinos, lors or mapsum
			if (mfold==0)write_raw<uint>("smap.raw",amap,F120_STride*F120_STride);
			else         write_raw<uint>("smap.raw",amap,MFold_XY*MFold_Z);
			//FILE *sh = fopen("smap.raw","wb");
			//fwrite(amap,sizeof(uint),F120_STride*F120_STride,sh);
			//fclose(sh);
			//printf("smap.raw written\n");
			//free(smap);
		}
		else printf("map file output suppresed\n");

		if (opt.isset("mapsum")){
			int n = 0;
			if (mfold==0) {
				for (int offset=0; offset<F120_STride*F120_STride; offset+=F120_STride) {
					for (int k=0; k<F120_STride; k++) {
						mapsum[n] += amap[offset+k];
						mapsum[F120_STride+k] += amap[offset+k];
						mapsum[F120_STride*2+k] += amap[offset+k];
						mapsum[F120_STride*2+n] += amap[offset+k];
					}
					n++;
				}
				write_raw<uint>("mapsum.raw",mapsum,F120_STride*3);

				if (F120_TrueNZ != F120_NZ){  // this for testing
					for (int k=0; k<F120_STride*3; k++)mapsum[k] = 0;
					int stride2 = F120_TrueNZ*F120_NXY;
					int z_start = (F120_NZ-F120_TrueNZ)/2;  // (96-48)/2 = 24
					int z_end = z_start+F120_TrueNZ;
					for (int z1=z_start; z1<z_end; z1++) for (int c1=0; c1<F120_NXY; c1++) for (int z2=z_start; z2<z_end; z2++) for (int c2=0; c2<F120_NXY; c2++){
						int n1 = (z1-z_start)*F120_NXY+c1;
						int k1 = (z2-z_start)*F120_NXY+c2;
						int n2 = z1*F120_NXY+c1;
						int k2 = z2*F120_NXY+c2;
						mapsum[n1]           += amap[n2*F120_STride+k2];
						mapsum[stride2+k1]   += amap[n2*F120_STride+k2];
						mapsum[stride2*2+k1] += amap[n2*F120_STride+k2];
						mapsum[stride2*2+n1] += amap[n2*F120_STride+k2];
					}
					write_raw<uint>("mapsum2.raw",mapsum,stride2*3);
				}
			}
			else {
				for (int offset=0; offset<MFold_XY*MFold_Z; offset += MFold_XY) {
					for (int k=0; k<MFold_XY; k++) {
						mapsum[n] += amap[offset+k];
						int z = mfold+k/F120_NXY;
						int c = k%F120_NXY;
						mapsum[F120_STride+z*F120_NXY+c] += amap[offset+k];
						mapsum[F120_STride*2+z*F120_NXY+c] += amap[offset+k];
						mapsum[F120_STride*2+n] += amap[offset+k];
					}
					n++;
				}
				write_raw<uint>("mapsum.raw",mapsum,F120_STride*3);
			}         
			free(mapsum);
		}      
		cudaFree(dev_amap);
	}

	if (opt.isset("compact") || opt.isset("minigen")){
		size_t ll=0;
		trys[3] = sum_map(amap,ll,mfold);   // this is final yield of good hits
		trys[4] = (double)ll;               // this is number of lors
		sprintf(name,"gentrys_%d_%d_%d.raw",s.nvox.x,s.nvox.y,s.nvox.z);
		write_raw<double>(name,trys,5);
	}

	if (amap) free(amap);

	free(bigbuf);
	cudaFree(dev_bigbuf);
	free(corners);
	cudaFree(dev_corners);
	free(gentrys);
	cudaFree(dev_gentrys);

	printf("IO time %.3f secs\n",timePassed(IO_StartingTime));
	return 0;
}

int save_map_as_phantom(char *name,uint *amap,int mfold)
{
	float *pmap = mycalloc<float>(F120_DZstride*F120_DCsize*F120_NXY,"pmap");
	if (!pmap) return 1;
	
	//int lowz1 = 0;
	int highz1 = F120_NZ;
	int lowz2= 0;
	//int highz2 = F120_NZ;
	if (mfold){
		highz1 = mfold+2;  // i.e z1 in [0, 48]
		lowz2 =  mfold;    // i.e z2 in [47,95]
	}
	int k = 0;
	for (int z1=0; z1<highz1; z1++) for (int c1=0; c1<F120_NXY; c1++) for (int z2=lowz2; z2<F120_NZ; z2++) for (int c2=0; c2<F120_NXY; c2++) {
		uint hits = amap[k++];
		if (hits > 0) {
			quad p = { z1,c1,z2,c2 };
			proper_lor(p);
			p.z1 -= 24;
			p.z2 -= 24;
			if (p.z1 >= 0 && p.z2 < 48){  // inside small detector
				int dc = abs(p.c2-p.c1);
				if (c1 > c2) dc = F120_NXY-dc;  // fix logically negative dc values
				dc -= F120_DCmin;
				if (dc >=0 && dc < F120_DCsize){  // in range 0-144
					int dz = p.z2-p.z1;
					int zsm_offset = (dz*(97-dz))/2;
					pmap[(dc*F120_NXY+p.c1)*F120_DZstride+zsm_offset+p.z1] += hits;
				}
			}
		}
	}

	write_raw<float>(name,pmap,F120_DZstride*F120_DCsize*F120_NXY);
	free(pmap);

	return 0;
}

double sum_map(uint *map,size_t &lors,int mfold)
{
	lors = 0;
	double lsum = 0.0;
	int stride = F120_STride;
	if (mfold > 0) stride =(F120_NZ-mfold)*F120_NXY;
	for (int k=0; k<stride*stride; k++) if (map[k]> 0){
		lors++;
		lsum += map[k];
	}

	return lsum;
}

int simple_lor_file(AOptions &opt,uint *map,Scanner &scn,Source &s,int mfold,FILE *logfile,double trys,int compact,int cull)
{
	size_t lors = 0;
	double lsum = 0.0;
	int stride = F120_STride;
	if (mfold > 0) stride =(F120_NZ-mfold)*F120_NXY;
	for (int k=0; k<stride*stride; k++) if (map[k]> 0){
		lors++;
		lsum += map[k];
	}

	if (lors <= 10) { printf("Too few lors found\n"); return 1; }
	trys /= lsum;
	float norm = (float)(1.0e06/lsum);
	printf("%d lor points found, total of %.0f counts, trys per count %.6f norm factor %.5e\n",lors,lsum,trys,norm);
	

	Lor *ltab = (Lor *)mycalloc<Lor>(lors,"ltab"); if (!ltab) return 1;

	int ngot = 0;
	if (mfold==0){
		for (int z1 = 0; z1<F120_NZ; z1++) for (int c1=0; c1<F120_NXY; c1++) for (int z2=0; z2<F120_NZ; z2++) for (int c2 = 0; c2<F120_NXY; c2++){
			int k = index_from_lor(z1,c1,z2,c2);
			if (map[k] > 0){
				if (ngot >= lors){ printf("overflow lors = ngot =%d at k = %d\n",lors,k); return 1; }
				ltab[ngot].key = key_from(z1,c1,z2,c2);
				ltab[ngot].val += (float)map[k]*norm;
				ngot++;
			}
		}
	}
	else {
		int z1max = mfold+2;   // add 1 so we can use < in for loop
		int z2min = mfold;
		for (int z1 = 0; z1<z1max; z1++) for (int c1=0; c1<F120_NXY; c1++) for (int z2=z2min; z2<F120_NZ; z2++) for (int c2 = 0; c2<F120_NXY; c2++){
			//if (z1==z2 && c2 < c1) ??
			int k = index_from_lor(z1,c1,z2,c2,mfold);
			if (map[k] > 0){
				if (ngot >= lors){ printf("overflow lors = ngot =%d at k = %d\n",lors,k); return 1; }
				ltab[ngot].key = key_from(z1,c1,z2,c2);
				ltab[ngot].val += (float)map[k]*norm;
				ngot++;
			}
		}
	}

	lors = ngot;
	fprintf(logfile," %d simple lors found, total of %.0f counts, norm factor %.5f mfold %d",lors,lsum,norm,mfold);
	printf(" %d simple lors found, total of %.0f counts, norm factor %.5f mfor %d\n",lors,lsum,norm,mfold);
	//printf("unique lors = %d, total = %d\n",ngot,lors);
	
	// NB header added to lor files 13/11/17  breaks older code
	NLtabheader nlhead = {lors,norm};

	char name[256];
	if (compact){
		//TODO compress using syms & discard small values
		sprintf(name,"nltab_%d_%d_%d.raw",s.nvox.x,s.nvox.y,s.nvox.z);
	 
		write_raw<NLtabheader>(name,&nlhead,1);
		append_raw<Lor>(name,ltab,lors);
		//if (cull) {
			//double tcut = 0.99;
			//if (opt == 1 && argc > 6)    tcut = atoi(argv[6]) / 100.0f;
			//else if (opt==4 && argc > 5) tcut = atoi(argv[5]) / 100.0f;
			//double cut = 1.0e06*(1.0 - tcut);
			//new_cull(ltab,lors,cut,tcut)
		//}
	}
	else{
		write_raw<NLtabheader>("nltab.raw",&nlhead,1);
		append_raw_quiet<Lor>("nltab.raw",ltab,lors);


		std::sort<Lor *>(ltab,ltab+lors,lorsort);  // C++ library - no need to use explicit vector type!
		write_raw<NLtabheader>("nltab_sorted.raw",&nlhead,1);
		append_raw_quiet<Lor>("nltab_sorted.raw",ltab,lors);
	}
	int sum99 = 0;
	int sum999 = 0;
	lsum = 0.0;
	for (int k=0; k<lors; k++){
		lsum += ltab[k].val;
		if (lsum > 990000.0 && sum99 == 0) sum99 = k;
		if (lsum > 999000 && sum999 == 0) {
			sum999 = k;
			break;
		}
	}
	if (opt.isset("allsorts")){
		sprintf(name,"nltab_keysort_%d_%d_%d.raw",s.nvox.x,s.nvox.y,s.nvox.z);
		std::sort<Lor *>(ltab,ltab+lors,lorsort_key);  // C++ library - no need to use explicit vector type!
		write_raw<NLtabheader>(name,&nlhead,1);
		append_raw_quiet<Lor>(name,ltab,lors);

		std::sort<Lor *>(ltab,ltab+lors,lorsort_z1z2);
		sprintf(name,"nltab_z1z2sort_%d_%d_%d.raw",s.nvox.x,s.nvox.y,s.nvox.z);
		write_raw<NLtabheader>(name,&nlhead,1);
		append_raw_quiet<Lor>(name,ltab,lors);
	}

	printf(         " %d lors in table of which 10k at %d and 1k after %d. Total normalised to 1.0^6",lors,sum99,sum999);
	fprintf(logfile," %d lors in table of which 10k at %d and 1k after %d. Total normalised to 1.0^6",lors,sum99,sum999);

	return 0;
}

cudaError_t gen_candidates(float *buf,float *dev_buf)
{
	
	return (cudaError_t)0;

}

// returns times passed since input argument initialised and this call
double timePassed(LARGE_INTEGER &StartingTime)
{
	LARGE_INTEGER EndingTime;
	QueryPerformanceCounter(&EndingTime);
	LARGE_INTEGER Frequency;
	LARGE_INTEGER ElapsedMicroseconds;
	QueryPerformanceFrequency(&Frequency);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	double timesec = 0.000001*ElapsedMicroseconds.QuadPart;
	return timesec;  

}

int block_hits(block &bx,ray &r)
{
	// block bx(-9.552f,9.552f,73.602f,83.602f,0.0f,76.416f);
	bx.hits = 0;
	float *lamsave = (float *)calloc(6,sizeof(float));
	float *lamall = (float *)calloc(6,sizeof(float));
	float4 hit;
	// loop for evental share by 6 threads
	for (int c_id = 0; c_id<6; c_id++){
		int r_id = c_id/2;
		float lam = (bx.c[c_id] - get_comp(&r.p,r_id)) / get_comp(&r.n,r_id);
		lamall[c_id] = lam;
		if (lam >0.0f){
			hit =r.p+lam*r.n;
			if (hit.x >= bx.c[0] && hit.x <= bx.c[1] &&
				hit.y >= bx.c[2] && hit.y <= bx.c[3] &&
				hit.z >= bx.c[4] && hit.z <= bx.c[5]) {
				lamsave[c_id] = lam;
				if (bx.hits <2){
					bx.lam[bx.hits] = lam;
					bx.face[bx.hits] = c_id;
					bx.hits++;
				}
			}
		}
	}
	printf("lama %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",lamall[0],lamall[1],lamall[2],lamall[3],lamall[4],lamall[5]);
	printf("lams %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",lamsave[0],lamsave[1],lamsave[2],lamsave[3],lamsave[4],lamsave[5]);
	if (bx.hits == 2 && bx.lam[0] > bx.lam[1]){
		float t = bx.lam[0];
		int f = bx.face[0];
		bx.lam[0] = bx.lam[1];
		bx.lam[1] =t;
		bx.face[0] = bx.face[1];
		bx.face[1] = f;
	}
	if (bx.hits >0){
		float4 hit1 = r.p + bx.lam[0]*r.n;
		printf("Entry hit face %d (%8.3f %8.3f %8.3f)",bx.face[0],hit1.x,hit1.y,hit1.z);
	}
	if (bx.hits >1){
		float4 hit2 = r.p + bx.lam[1]*r.n;
		printf("  Exit hit face %d (%8.3f %8.3f %8.3f)",bx.face[1],hit2.x,hit2.y,hit2.z);
	}
	if (bx.hits>0)printf("\n");

	free(lamsave);
	free(lamall);

	return 0;
}

// mimic F120 3d sinogram code from normsim expect ushorts or uints or reals
template <typename T> int bin_sino3d(T *s,T *ssrb,int c1,int z1,int c2,int z2,T hits,Scanner &scn)
{
	int uflip = 1;
	int roll = F120_NXY/4-6;
	int rdcut = F120_NZ-1;

	static int first_call = 1;
	// this code F120 span 3 specific. Need to generalize??
	// NSegs =2*(F120_NZ)/3)+1;  => 33 for nz=48 & 65 for NZ=96
	static int segbase[F120_NSegs];
	static int zstart[F120_NSegs];
	if (first_call){
		int step = F120_NZ*2-1;
		segbase[0] = 0;
		segbase[1] = segbase[0] + step;
		step -= 4;
		segbase[2] = segbase[1] + step;
		segbase[3] = segbase[2] + step;
		for (int i=3; i<F120_NSegs-2; i+=2) {   // this was 31 NB i=3 pass partly redundant
			segbase[i] = segbase[i-1]+step;
			step -= 6;
			segbase[i+1] = segbase[i] + step;
		}
		segbase[F120_NSegs-2] = segbase[F120_NSegs-3]+step;
		segbase[F120_NSegs-1] = segbase[F120_NSegs-2]+1;
		zstart[0] = 0;
		zstart[1] = zstart[2] = 2;
		for (int i=3; i<F120_NSegs; i+=2)	zstart[i] = zstart[i+1] = zstart[i-2]+3;

		//for(int i=0;i<33;i++)printf("seg %2d base %4d zstart %d\n",i,segbase[i],zstart[i]);
		first_call=0;
	}

	int flip=1;
	int u,phi;
	if (abs(z1-z2)> rdcut) return 0;


	c1 = (c1+roll)%F120_NXY;
	c2 = (c2+roll)%F120_NXY;

	//phi =   (((c1+c2)%F120_NXY)/2);  // same phi for odd and even interlace
	phi =   (((c1+c2+1)%F120_NXY)/2);  // change Aug 30 2011, phi=0 between 287 and 0

	// this logic give F120 segment allocations - Aug 31 2011
	if (c1>phi && c1 <= (phi+F120_NXY/2)){
		if (z1 > z2) flip = 0;
		else flip = 1;
	}
	else{
		if (z2 > z1) flip =0;
		else flip =1;
	}

	int ca = (c1 > phi) ? c1 : c1+F120_NXY;
	int cb = (c2 > phi) ? c2 : c2+F120_NXY;

	u = abs(ca-cb) + (F120_SNX - 1)/2 - F120_NXY/2;

	// having got lookup tables, segment handling very simple.
	int seg = 0;
	int dz = abs(z1-z2);
	if (dz < 2) seg = 0;
	else if (dz >= F120_NZ-1) seg = 31;
	else seg = 2*((dz+1)/3)-1;
	if (flip && seg>0) seg += 1;
	int zpos = segbase[seg] + z1+z2 - zstart[seg];

	if (uflip) u = F120_SNX-u-1;            //flip u

	if (u>=0 && u<F120_SNX) {
		s[(zpos*F120_SNY+phi)*F120_SNX+u] += hits;
		//s[zpos][phi][u]++;
		if (dz<24) ssrb[((z1+z2)*F120_SNY+phi)*F120_SNX+u] += hits;
		//if(dz<24) ssrb[z1+z2][phi][u]++;
		//if(fbug && zpos==96)fprintf(fbug," %3d %3d %3d %3d %2d %2d\n",c1,c2,phi,u,z1,z2);
		//if(fbug2 && zpos==187)fprintf(fbug2," %3d %3d %3d %3d %2d %2d\n",c1,c2,phi,u,z1,z2);
	}
	return 0;
}

int debug_sino(void)
{
	// this code F120 span 3 specific. Need to generalize??
	// NSegs =2*(F120_NZ)/3)+1;  => 33 for nz=48 & 65 for NZ=96
	int segbase[F120_NSegs];
	int zstart[F120_NSegs];

	int step = F120_NZ*2-1;
	segbase[0] = 0;
	segbase[1] = segbase[0] + step;
	step -= 4;
	segbase[2] = segbase[1] + step;
	segbase[3] = segbase[2] + step;
	for (int i=3; i<F120_NSegs-2; i+=2) {   // this was 31
		segbase[i] = segbase[i-1]+step;
		step -= 6;
		segbase[i+1] = segbase[i] + step;
	}
	segbase[F120_NSegs-2] = segbase[F120_NSegs-3]+step;
	segbase[F120_NSegs-1] = segbase[F120_NSegs-2]+1;
	zstart[0] = 0;
	zstart[1] = zstart[2] = 2;
	for (int i=3; i<F120_NSegs; i+=2)	zstart[i] = zstart[i+1] = zstart[i-2]+3;
	printf("sino has %d segments\n",F120_NSegs);
	for(int i=0;i<F120_NSegs;i++)printf("seg %2d base %4d zstart %d\n",i,segbase[i],zstart[i]);

	return 0;
}

int append_existing(uint *amap, uint *dev_amap, Source &s, int mfold, char *root)
{
	printf("append existing called\n");
	char name[256];	
	path_from_root(name,root,s.nvox);
	
	NLtabheader nlhead;
	if(read_raw_quiet<NLtabheader>(name,&nlhead,1)) return 1;

	size_t lors = nlhead.lors;
	Lor *ltab = mymalloc<Lor>(lors);
	if(read_raw_skip<Lor>(name,ltab,lors,sizeof(NLtabheader))) return 1;

	printf("file %s has %d lors, norm is %f\n",name,lors,nlhead.norm);

	if (mfold != 0 && mfold != 47) {printf("ONLY mfold = %d but only 0 or 47 are allowd\n",mfold); return 1;}

	int MFold_XY = F120_STride;
	int MFold_Z =  F120_STride;
	
	if (mfold > 0){
		MFold_XY = (F120_NZ-mfold)*F120_NXY;  //z2 c2 range    mfold <= z2 < NZ and z2 >= z1 (wlg by lor end sorting)
		MFold_Z = (mfold+2)*F120_NXY;         // z1 c1 range   0 <= z1 <= mfold+1      
	}

	size_t mlen = MFold_XY*MFold_Z;

	quad p;
	double lsum = 0.0;
	for (int k=0; k<lors; k++){
		lor_from(ltab[k].key,p);
		int z1 = p.z1;
		int z2 = p.z2;
		if (mfold){
			z1 = min(z1,mfold+1);
			z2 = max(0,z2-mfold);
		}
		int index = (z1*F120_NXY+p.c1)*MFold_XY+(z2*F120_NXY)+p.c2;
		amap[index] += (uint)(0.5f + ltab[k].val/nlhead.norm);
		lsum += ltab[k].val/nlhead.norm;
	}

	cudaError_t cudaStatus = cudaMemcpy(dev_amap,amap,mlen*sizeof(uint),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy in append_existing failed [%s]",name,cudaGetErrorString(cudaStatus));  return 1; }


	printf("added %d lors from %s with total of %.2f counts\n",lors,name,lsum);
	return 0;
}

void path_from_root(char *dest, char *root,int3 p)
{
	if (p.x <74)       sprintf(dest,"%s\\seta\\nltab_%2d_%2d_%2d.raw",root,p.x,p.y,p.z);
	else if (p.x <84)  sprintf(dest,"%s\\setb\\nltab_%2d_%2d_%2d.raw",root,p.x,p.y,p.z);
	else if (p.x <94)  sprintf(dest,"%s\\setc\\nltab_%2d_%2d_%2d.raw",root,p.x,p.y,p.z);
	else if (p.x <104) sprintf(dest,"%s\\setd\\nltab_%2d_%2d_%2d.raw",root,p.x,p.y,p.z);
	else if (p.x <114) sprintf(dest,"%s\\sete\\nltab_%2d_%2d_%2d.raw",root,p.x,p.y,p.z);
	else               sprintf(dest,"%s\\setf\\nltab_%2d_%2d_%2d.raw",root,p.x,p.y,p.z);

	printf("path_to_root: %s\n",dest);
	return;
}

int set_F120(Scanner &scn)
{
	// define F120 scanner as default.  Add other constructors as required
	scn.Rmin = 73.602000f;
	scn.Rmax = 84.145916f;
	scn.Rsize = 10.000000f;
	scn.Csize = 1.592000f;
	scn.Cnum = 12;
	scn.BPnum = 24;
	scn.BZnum = 4;
	scn.NXY = 288;
	scn.NZ = 48;
	scn.NDoi = 16;
	scn.STride = (scn.NXY*scn.NZ);
	scn.BPhi = 0.261799f;
	scn.BPface = 19.104000f;
	scn.BZface = 76.416000f;
	scn.BPfullface = 19.379774f;
	scn.Thetacut = 0.478823f;
	scn.FCut1 = 74.219237f;
	scn.FCut2 = 84.145916f;
	scn.LSOattn = 0.087000f;
	//LSOattn_Recip 0.087000f
	scn.H2Oattn = 0.009310f;
	//#define F120_XYBin 0.865759f  // this from header
	// this for max square within inner radius
	scn.XYBin = 0.813194f;
	scn.ZBin = 0.796f;
	scn.NXYbins = 128;
	scn.NZbins = 95;

	// these for sinograms
	scn.SNX  = 128;
	scn.SNY  = 144;
	scn.SNZ  = 1567;
	scn.SNZ2  = 95;
	//printf("F120 settings used for Scanner\n");
	return 0;
}

