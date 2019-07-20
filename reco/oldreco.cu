// reco based on template.cu

// cuda stuff
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h> 
#include <helper_math.h>
#include <device_functions.h>

// generic stuff
#include "mystuff.h"
#include "aoptions.h"

// pet stuff
#include "F120_long_defs.h"
#include "cudapet.h"
#include "lors.h"
#include "smatrix.h"
#include "reco.h"
#include "reco_kernels.h" 

// windows stuff
#include <algorithm>
#include <Windows.h>


int roibox_cut(int x, int y);
int compute_forward(SMfull &sm, float *voxvals, float *tsum);
int bigjob(SMfull &sm,AOptions &opt);
double timePassed(LARGE_INTEGER &StartingTime);
int show_full_tsum(float *tsum);
int simple_check(SMfull &sm,AOptions &opt,float *tsum,int nx,int ny,int sector);
int dump_sm(SMfull &sm,AOptions &opt);
int do_forward_projection(SMfull &sm,AOptions &opt,float *voxval,float *tsum,float *zdzmap);
int cuda_do_forward_projection(SMfull &sm,AOptions &opt,float *voxval,float *tsum,float *zdzmap);
int   cyl_fill(float *vox,AOptions &opt,int nxy,int nz,double dxy);
int const_fill(float *vox,AOptions &opt,int nxy,int nz,float val);
int setup_cuda_sm(SMfull &sm,cudaSM &host_sm,cudaSM **dev_sm_out);
int setup_cuda_vmap(VoxMap &vm);
template <typename T> int swizzle_buffer(T *a,int n1,int n2,int n3,int m1,int m2,int m3);
int map_swizz(char *name_in, char *name_out);
int map_chop(char *name_in, char *name_out);
int make_dc(int c1,int c2);
int host_do_forward_project(SMfull &sm,float *voxval,float *zdzmap);
int do_mlem(SMfull&sm, AOptions &opt);
template <typename T> int make_buffers(T **buf_out,T **dev_buf_out, size_t len, char *tag);
template <typename T> int read_buffers(char *name, int len, T *h_buf, T *d_buf,T rescale);
template <typename T> int copy_buffer_to(int len, T *h_buf, T *d_buf);
template <typename T> int copy_buffer_from(int len, T *h_buf, T *d_buf);
template <typename T> int clear_buffer(int len, T *d_buf);
int cyl_buffer_fill_normalized(float *vox,double val);
int do_forward(SMfull &sm,char *vol_in,char *zmap_out);
int do_backward(SMfull &sm,char *zmap_in,char *vol_out);

char *smfile ="D:\\data\\syseff.raw";
char *tvfile ="D:\\data\\big_tveff.raw";       //changed back  13/11/17

char *mini_smfile ="D:\\data\\smbigger.raw";
//char *mini_tvfile ="D:\\data\\tvbigger2.raw";   //changed 01/11/17
char *mini_tvfile ="D:\\data\\tvbigger_szk.raw";  //changed 29/11/17

int main(int argc, char *argv[])
{
	LARGE_INTEGER StartingTime;
	QueryPerformanceCounter(&StartingTime);

	if (argc < 2){
		printf("Reco - PET reconstuction with complete system matrix\n",F120_ID);
		printf("sysmat:filename                   system matix (default %s)\n",smfile);
		printf("mlem                              MLEM reco, same as OSEM:1\n");
		printf("osem:n                            OSEM n subsets, n=1,2,4 or 8 supported\n");
		printf("maxit:n [24/n]                    Max full OMEM passes\n");
		printf("dzcut:val                         max dz [47 i.e. all]\n");
		printf("sector:val                        use single sector [-1 i.e, all]\n");
		printf("one_x:sx                          use single voxel sx sy [default use all]\n");
		printf("one_y:sy                          set sy [defaults to sx which must be set]\n");
		printf("cylfill                           use cylinder for activity\n");
		printf("cylrad                            cylinder radius [23.0]\n");
		printf("cyltest or cyltestpr              write active volume and exit, add pr to print\n");
		printf("ones                              file ROI with 1.0f\n");
		printf("voxval                            write active volume\n");
		printf("tsum                              write full tsum dataset\n");
		printf("ones                              fill ROI voxels with 1.0f\n");
		printf("mapsum                            write tsum summed over slices\n");
		printf("cuda                              use cuda!\n");
		printf("minivox                              use small voxel defaults\n");
		printf("mapswizz <fin> <fout>             convert full map to swizz form\n");
		printf("mapchop  <fin> <fout>             extract mid 48 z-range from 96 map\n");
		printf("doforward <vol in> <zdzmap out>   do one forward projection\n");
		printf("dobackward <zdzmap in> <vol out>  do one backward projection\n");

		return 0;
	}
	
	char cudalog_name[] = "D:\\logfiles\\cudareco.log";
	FILE *logfile = fopen(cudalog_name,"a");
	if (!logfile) {
		logfile = fopen(cudalog_name,"w");
		if (logfile) printf("new %s logfile created\n",cudalog_name);
	}
	if (!logfile) { printf("can't open %s",cudalog_name);  return 1; }
	fprintf(logfile,"cudareco %s version 2.0 args: ",F120_ID); 
	for (int k=0; k<argc; k++) fprintf(logfile," %s",argv[k]); 
	fprintf(logfile,"\n");

	AOptions opt(argc,argv,1);

	// misc quick options here before open system matrix
	if (opt.isset("mapchop")){
		int k= opt.isset("mapchop");
		return map_chop(argv[k+1], argv[k+2]);
	}

	if (opt.isset("mapswizz")){
		int k= opt.isset("mapswizz");
		return map_swizz(argv[k+1], argv[k+2]);
	}


	char sm_name[256];
	if (opt.isset("sysmat"))       strcpy(sm_name, opt.get_string("sysmat"));
	else if (opt.isset("minivox")) strcpy(sm_name,mini_smfile);
	else                           strcpy(sm_name,smfile);
	SMfull sm(sm_name);
	if (sm.numlors <10) return 1;
	
	if (opt.isset("doforward"))
	{
		int k=opt.isset("doforward");
		return do_forward(sm,argv[k+1],argv[k+2]);
	}

	if (opt.isset("dobackward"))
	{
		int k=opt.isset("dobackward");
		return do_backward(sm,argv[k+1],argv[k+2]);
	}

	if (opt.isset("mlem") || opt.isset("osem")){
		return do_mlem(sm,opt);
	}

	if (opt.isset("cudatest")) {
		cudaSM *dev_sm;
		cudaSM host_sm;
		setup_cuda_sm(sm,host_sm,&dev_sm);
		check_sm<<<1,16>>>(dev_sm);
		return 0;
	}
	//printf("A\n");

	if (opt.isset("dump")) dump_sm(sm,opt);
	else bigjob(sm,opt);
	
	if (opt.isset("cuda")){
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess)	{ printf("Failed to deinitialize the cuda device?? error=%s\n",cudaGetErrorString(cudaStatus)); }
	}

	printf("Total time %.3f secs\n",timePassed(StartingTime));
	fclose(logfile);
	return 0;
}

int do_forward(SMfull &sm,char *vol_in,char *zmap_out)
{
	LARGE_INTEGER CudaTime;
	QueryPerformanceCounter(&CudaTime);

	printf("Running MLEM!\n");
	//system matrix
	cudaSM *dev_sm = NULL;
	cudaSM host_sm;
	VoxMap vm;
	cudaError_t cudaStatus;

	int nlors = F120_NXY *     F120_DCsize *  F120_DZstride;  //zdz format
	int nvox =  F120_NXYbins * F120_NXYbins * F120_NZbins;

	// measured actvity  (#lors)  init from external file
	float *vol = NULL;
	float *dev_vol = NULL;
	if (make_buffers<float>(&vol,&dev_vol,nvox,"vol")) return 1;
	if (read_buffers<float>(vol_in,nvox,vol,dev_vol,1.0f)) return 1;

	// forward projection (#lors) int with zeros
	float *zdzmap = NULL;
	float *dev_zdzmap = NULL;
	if (make_buffers<float>(&zdzmap,&dev_zdzmap,nlors,"zdzmap")) return 1;

	// efficiency sums (#voxels)  init from external file
	float *teffs = NULL;
	float *dev_teffs = NULL;
	if (make_buffers<float>(&teffs,&dev_teffs,nvox,"teffs")) return 1;
	if (read_buffers<float>(tvfile,nvox,teffs,dev_teffs,1.0f)) return 1;

	// sm for device
	if (setup_cuda_sm(sm,host_sm,&dev_sm)) return 1;
	if (setup_cuda_vmap(vm)) return 1;

	// one fp step
	for (int kv=0; kv<sm.voxels; kv++){
		// current activity => lor lors
	   //forward_project_faster<<<64,64>>>(dev_sm,dev_vol,dev_zdzmap,dev_teffs,kv,0);  //TODO fix this for osem
	   // forward_project_faster<<<64,64>>>(dev_sm,dev_vol,dev_zdzmap,dev_teffs,kv,1);
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"forward_proj kernel error: [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }

	printf("forward projection time %.3f secs\n",timePassed(CudaTime));

	cudaDeviceSynchronize();
	if (copy_buffer_from<float>(nlors,zdzmap,dev_zdzmap))return 1;
	write_raw<float>(zmap_out,zdzmap,nlors);

	if (vol) free(vol);
	if (dev_vol) cudaFree(dev_vol);
	if (zdzmap) free(zdzmap);
	if (dev_zdzmap) cudaFree(dev_zdzmap);

	return 0;
}

int do_backward(SMfull &sm,char *zmap_in,char *vol_out)
{
	LARGE_INTEGER CudaTime;
	QueryPerformanceCounter(&CudaTime);

	printf("Running MLEM!\n");
	//system matrix
	cudaSM *dev_sm = NULL;
	cudaSM host_sm;
	VoxMap vm;
	cudaError_t cudaStatus;

	int nlors = F120_NXY *     F120_DCsize *  F120_DZstride;  //zdz format
	int nvox =  F120_NXYbins * F120_NXYbins * F120_NZbins;

	// measured actvity  (#lors)  init from external file
	float *vol = NULL;
	float *dev_vol = NULL;
	if (make_buffers<float>(&vol,&dev_vol,nvox,"vol")) return 1;

	// forward projection (#lors) int with zeros
	float *zdzmap = NULL;
	float *dev_zdzmap = NULL;
	if (make_buffers<float>(&zdzmap,&dev_zdzmap,nlors,"zdzmap")) return 1;
	if (read_buffers<float>(zmap_in,nlors,zdzmap,dev_zdzmap,1.0f)) return 1;

	// efficiency sums (#voxels)  init from external file
	float *teffs = NULL;
	float *dev_teffs = NULL;
	if (make_buffers<float>(&teffs,&dev_teffs,nvox,"teffs")) return 1;
	if (read_buffers<float>(tvfile,nvox,teffs,dev_teffs,1.0f)) return 1;


	// sm for device
	if (setup_cuda_sm(sm,host_sm,&dev_sm)) return 1;
	if (setup_cuda_vmap(vm)) return 1;

	// one fp step
	for (int kv = 0; kv<sm.voxels; kv++){
  //      backward_project_faster<<<64,64>>>(dev_sm,dev_vol,dev_zdzmap,dev_teffs,kv);  // TODO fix this for OSEM
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"backward_proj kernel error: [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }

	printf("backward projection time %.3f secs\n",timePassed(CudaTime));

	cudaDeviceSynchronize();
	if (copy_buffer_from<float>(nvox,vol,dev_vol))return 1;
	write_raw<float>(vol_out,vol,nvox);

	if (vol) free(vol);
	if (dev_vol) cudaFree(dev_vol);
	if (zdzmap) free(zdzmap);
	if (dev_zdzmap) cudaFree(dev_zdzmap);
	
	return 0;
}

// osem added 28/10/17
int do_mlem(SMfull &sm, AOptions &opt)
{
	LARGE_INTEGER CudaTime;
	LARGE_INTEGER FpTime;    float Fpsum = 0.0f;
	LARGE_INTEGER BpTime;    float Bpsum = 0.0f;
	LARGE_INTEGER VfTime;    //float Vfsum = 0.0f;
	LARGE_INTEGER LfTime;    //float Lfsum = 0.0f;
	LARGE_INTEGER RunTime;

	QueryPerformanceCounter(&CudaTime);
	QueryPerformanceCounter(&RunTime);

	int osem = opt.set_from_opt("osem",1);
	//if (osem !=1){ printf("sorry osem is broken at the moment - using mlem\n"); osem = 1; }
	if(osem==1)printf("Running MLEM!\n");
	else printf("Running OSEM %d subsets!\n",osem);
	//system matrix
	cudaSM *dev_sm = NULL;  
	cudaSM host_sm;
	VoxMap vm;
	cudaError_t cudaStatus;


	int nlors = F120_NXY *     F120_DCsize *  F120_DZstride;
	//int nvox =  F120_NXYbins * F120_NXYbins * F120_NZbins;    //this for cartesian 128*128*95
	int nvox =  F120_SZKsize;       //this for szk 8*95*1661
	int big_nvox = nvox*(1+2+4+8);  // TODO just store required subsets?

	// measured actvity  (#lors)  init from external file
	float *meas = NULL;
	float *dev_meas = NULL;
	if (make_buffers<float>(&meas,&dev_meas,nlors,"meas")) return 1;
	if (read_buffers<float>("measured.raw",nlors,meas,dev_meas,1.0f)) return 1;

	// forward projection (#lors) int with zeros
	float *fproj = NULL;
	float *dev_fproj = NULL;
	if (make_buffers<float>(&fproj,&dev_fproj,nlors,"fproj")) return 1;

	// backward projection (#voxels) int with zeros
	float *bproj = NULL;
	float *dev_bproj = NULL;
	if (make_buffers<float>(&bproj,&dev_bproj,nvox,"bproj")) return 1;

	// estimated activity (#voxels) init using measured lors (maybe use bp not uniform)
	float *act = NULL;
	float *dev_act = NULL;
	if (make_buffers<float>(&act,&dev_act,nvox,"act")) return 1;   // this clears dev buffer

	// efficiency sums (#voxels)  init from external file
	float *teffs = NULL;
	float *dev_teffs = NULL;
	if (make_buffers<float>(&teffs,&dev_teffs,big_nvox,"teffs")) return 1;
	if (opt.isset("minivox")){ if(read_buffers<float>(mini_tvfile,big_nvox,teffs,dev_teffs,1.0f)) return 1; }
	else if (read_buffers<float>(tvfile,big_nvox,teffs,dev_teffs,1.0f)) return 1;
	//for (int k=0; k<nvox; k++) teffs[k] = 1.0f;

	// sm for device
	if(setup_cuda_sm(sm,host_sm,&dev_sm)) return 1;   
	if(setup_cuda_vmap(vm)) return 1;   

	// check for existing roi_start file
	if (read_raw_quiet<float>("roi_start.raw", act, nvox)==0){
		copy_buffer_to<float>(nvox,act,dev_act);
		printf("activity initialised from roi_start.raw\n");
	}
	else{
		// use back projection for initialization of buffer instead of constant filling.
		QueryPerformanceCounter(&BpTime);
		for (int kv = 0; kv<sm.voxels; kv++){
			backward_project_faster<<<64,64>>>(dev_sm,dev_act,dev_meas,1,0,kv);  // back proj measured lors to activity.
		}
		cudaDeviceSynchronize();
		printf("initial bp time %.3f secs\n",timePassed(BpTime));
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"initial backward_proj kernel error: [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
		if(copy_buffer_from<float>(nvox,act,dev_act))return 1;
		write_raw<float>("roi_start.raw",act,nvox);
	}

	char name[256];
	int maxit = 24/osem;
	if (opt.isset("maxit")) maxit = opt.set_from_opt("maxit",maxit);

	char bugname[256];
	//return 0;

	for (int iter=0; iter<maxit; iter++) for (int osem_set = 0; osem_set<osem; osem_set++) {

		if (clear_buffer(nlors,dev_fproj)) return 1;
		QueryPerformanceCounter(&FpTime);
		for (int kv=0; kv<sm.voxels; kv++){
			// current activity => lor lors
			//forward_project_faster<64><<<64,64>>>(dev_sm,dev_act,dev_fproj,osem,osem_set,kv,0);
			forward_project_faster<128><<<64,128>>>(dev_sm,dev_act,dev_fproj,osem,osem_set,kv,0);
			//cudaStatus = cudaGetLastError();
			//if (cudaStatus != cudaSuccess) { fprintf(stderr,"forward_project kernel error it %d: [%s] kv %d even 0\n",iter,cudaGetErrorString(cudaStatus),kv); return 1; }
			//forward_project_faster<64><<<64,64>>>(dev_sm,dev_act,dev_fproj,osem,osem_set,kv,1);
			forward_project_faster<128><<<64,128>>>(dev_sm,dev_act,dev_fproj,osem,osem_set,kv,1);
			//cudaStatus = cudaGetLastError();
			//if (cudaStatus != cudaSuccess) { fprintf(stderr,"forward_project kernel error it %d: [%s] kv %d even 1\n",iter,cudaGetErrorString(cudaStatus),kv); return 1; }
		}
		cudaDeviceSynchronize();
		printf("fp time %.3f secs\n",timePassed(FpTime));
		Fpsum += (float)timePassed(FpTime);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"forward_project kernel error it %d: [%s]\n",iter,cudaGetErrorString(cudaStatus)); return 1; }

		if (0){
			cudaDeviceSynchronize();
			sprintf(bugname,"fpdone%2.2d.raw",iter);
			if (copy_buffer_from<float>(nlors,fproj,dev_fproj))return 1;
			cudaDeviceSynchronize();
			write_raw<float>(bugname,fproj,nlors);
		}

		QueryPerformanceCounter(&LfTime);
		lor_factors <<<F120_DCsize,256>>>(dev_meas,dev_fproj,osem,osem_set);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"lor_factors kernel error it %d: [%s]\n",iter,cudaGetErrorString(cudaStatus)); return 1; }
		//cudaDeviceSynchronize();
		//printf("lf time %.3f secs\n",timePassed(LfTime));
		// Lfsum += timePassed(LfTime);

		if (0){
			cudaDeviceSynchronize();
			sprintf(bugname,"lfdone%2.2d.raw",iter);
			if (copy_buffer_from<float>(nlors,fproj,dev_fproj))return 1;
			cudaDeviceSynchronize();
			write_raw<float>(bugname,fproj,nlors);
		}


		QueryPerformanceCounter(&BpTime);
		for (int kv = 0; kv<sm.voxels; kv++){
			backward_project_faster<<<64,64>>>(dev_sm,dev_bproj,dev_fproj,osem,osem_set,kv);
			//backward_project_faster2<<<64,64>>>(dev_sm,dev_bproj,dev_fproj,osem,osem_set,kv,0);  // back proj measured lors to activity.
			//backward_project_faster2<<<64,64>>>(dev_sm,dev_bproj,dev_fproj,osem,osem_set,kv,1);  // back proj measured lors to activity.

		}
		cudaDeviceSynchronize();
		printf("bp time %.3f secs\n",timePassed(BpTime));
		Bpsum += (float)timePassed(BpTime);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"backward_proj kernel error it %d: [%s]\n",iter,cudaGetErrorString(cudaStatus)); return 1; }

		if (0){
			cudaDeviceSynchronize();
			sprintf(bugname,"bpdone%2.2d.raw",iter);
			if (copy_buffer_from<float>(nvox,bproj,dev_bproj))return 1;
			cudaDeviceSynchronize();
			write_raw<float>(bugname,bproj,nvox);
		}

		QueryPerformanceCounter(&VfTime);
		vox_factors<<<128,256>>>(dev_teffs,dev_act,dev_bproj,osem,osem_set);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"vox_factors kernel error it %d: [%s]\n",iter,cudaGetErrorString(cudaStatus)); return 1; }
		//cudaDeviceSynchronize();
		//printf("vf time %.3f secs\n",timePassed(VfTime));
		// VFsum += timePassed(VfTime);

		clear_buffer<float>(nlors,dev_fproj);  // BUG fix 16/10/17
		clear_buffer<float>(nvox,dev_bproj);
		// thats it!

		cudaDeviceSynchronize();
		if (maxit < 6 || (iter+1)%5==0 || iter+1 == maxit){
			if (osem==1){
				sprintf(name,"mlem%2.2d.raw",iter+1);
				if (copy_buffer_from<float>(nvox,act,dev_act))return 1;
				write_raw<float>(name,act,nvox);
			}
			else if (osem_set==osem-1){
				sprintf(name,"osem%2.2d_subset%2.2d_iter%2.2d.raw",osem,osem_set+1,iter+1);
				if (copy_buffer_from<float>(nvox,act,dev_act))return 1;
				write_raw<float>(name,act,nvox);
			}
		}
	}
	if (meas) free(meas);
	if (dev_meas) cudaFree(dev_meas);
	if (fproj) free(fproj);
	if (dev_fproj) cudaFree(dev_fproj);
	if (bproj) free(bproj);
	if (dev_bproj) cudaFree(dev_bproj);
	if (act) free(act);
	if (dev_act) cudaFree(dev_act);
	if (teffs) free(teffs);
	if (dev_teffs) cudaFree(dev_teffs);

	printf("total times mlem %.3f,  fp %.3f, bp %.3f secs\n",timePassed(RunTime),Fpsum,Bpsum);
	return 0;
}

int setup_cuda_vmap(VoxMap &vm)
{
	// sector constants to global device memory
	int *map = vm.amap_x();   
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpyToSymbol(dev_map8_x,map,24*sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol map8_x failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	map = vm.amap_y();
	cudaStatus = cudaMemcpyToSymbol(dev_map8_y,map,24*sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol map8_y failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	map = vm.amap_c();
	cudaStatus = cudaMemcpyToSymbol(dev_map8_c,map,16*sizeof(int));
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol map8_c failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	return 0;
}

int setup_cuda_sm(SMfull &sm,cudaSM &host_sm,cudaSM **dev_sm_out)
{
	// allocate actual sm buffer on device
	cudaSM  *dev_sm = NULL;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_sm,sizeof(cudaSM));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc dev_sm failed [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }

	// mirror copy of dev_sm on host
	//cudaSM host_sm;
	host_sm.voxels = sm.voxels;
	cudaStatus = cudaMalloc((void**)&host_sm.v,sm.voxels*sizeof(SMfull_vox));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc for sm.voxels failed [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	// copy voxels to device pointer
	cudaStatus = cudaMemcpy(host_sm.v,sm.v,sm.voxels*sizeof(SMfull_vox),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy to sm.v failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	host_sm.numlors = sm.numlors;
	cudaStatus = cudaMalloc((void**)&host_sm.lors,sm.numlors*sizeof(smlor));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc for sm.lors failed [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	// copy lors to device pointer
	cudaStatus = cudaMemcpy(host_sm.lors,sm.lors,sm.numlors*sizeof(smlor),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy to sm.lors failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	// copy struct to device
	cudaStatus = cudaMemcpy(dev_sm,&host_sm,sizeof(cudaSM),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { printf("cudaMemcpy to dev_sm failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	*dev_sm_out = dev_sm;
	return 0;
}

int dump_sm(SMfull &sm,AOptions &opt)
{
	int nx = opt.set_from_opt("nx",64);
	int ny = opt.set_from_opt("ny",64);
	quad p;
	for (int kv = 0; kv < sm.voxels; kv++){
		if (sm.v[kv].nx != nx || sm.v[kv].ny != ny) continue;
		printf("found voxel %d %d at kv=%d with lors = %d\n",nx,ny,kv,sm.v[kv].lors);
		for (int kl = 0; kl < sm.v[kv].lors; kl++){
			uint key = sm.key(kv,kl);
			small_lor_from(key,p);
			printf("%5d (%2d %3d)-(%2d %3d) %9.5f %9.5f\n",kl,p.z1,p.c1,p.z2,p.c2,sm.val(kv,kl,0),sm.val(kv,kl,1));
		}
	}
	return 0;
}

double box_bit(double r,float3 &p)
{
	// return area under arc for x [p.x,p.y] within box base p.z
	double a = p.x;
	double b = p.y;
	double theta_a = asin(a/r);
	double theta_b = asin(b/r);
	double area = r*r*0.5*(theta_b - theta_a + 0.5*(sin(2.0*theta_b)-sin(2.0*theta_a)));
	area -= p.z*(b-a); // remove contibution below p.z

	return area;
}

double box_in_circle(double r,float4 &p)
{
	// float4 format {x0,y0,dx,dy}
	//                                           |
	// case 0  none inside                       |   b-----d
	// case 1  just (a) inside                   |   |     |
	// case 2  both (a) & (b) inside             |   |     |
	// case 3  both (a) & (c) inside             |   a-----c
	// case 4  (a) (b) & (c) inside              |
	// case 5  all inside 			             0-------------------

	float xa = p.x;
	float ya = p.y;
	double ra = sqrt(xa*xa+ya*ya);
	float xb = xa;
	float yb = ya+p.w;
	double rb = sqrt(xb*xb+yb*yb);
	float xc = xa+p.z;
	float yc = ya;
	double rc = sqrt(xc*xc+yc*yc);
	float xd = xc;
	float yd = yb;
	double rd =sqrt(xd*xd+yd*yd);

	if      (rd < r  ) return p.z*p.w;   //  inside:  easy case 5;
	else if (ra >=  r) return 0.0;       //  outside: easy case 0;

	else if (rb > r && rc > r) {        //   a inside: case 1

		float xh = (float)sqrt(r*r-ya*ya);
		float3 q ={ xa,xh,ya };
		return box_bit(r,q);
	}
	else if (rb < r && rc > r) {        //  a & b inside: case 2
		float xl = (float)sqrt(r*r-yb*yb);
		float xh = (float)sqrt(r*r-ya*ya);
		float3 q ={ xl,xh,ya };
		return box_bit(r,q)+(xl-xa)*(yb-ya);
	}
	else if (rb > r && rc < r) {        // a & c inside: case 3
		float3 q ={ xa,xc,ya };
		return box_bit(r,q);
	}
	else if (rb < r && rc < r) {       // a, b & c inside: case 4
		float xl = (float)sqrt(r*r-yb*yb);
		float3 q ={ xl,xc,ya };
		return box_bit(r,q) +(xl-xa)*(yb-ya);
	}
	else printf("unexpected case in box_in_circle p %f %f %f r %f\n",p.x,p.y,p.z,r);
	return 0.0;

}

int cyl_fill(float *vox,AOptions &opt,int nxy,int nz,double dxy)
{
	double r = opt.set_from_opt("cylrad",F120_XYBin*nxy);
	printf("cyl_fill for radius %8.3f nxy %d nz %d\n",r,nxy,nz);
	int stride = nxy*nxy;
	for (int k=0; k<stride*nz;k++) vox[k] = 0;

	int mx = nxy/2;
	int my = nxy/2;
	float4 p = { 0.0f,0.0f,(float)dxy,(float)dxy };
	for (int kx=0; kx<nxy/2; kx++) {
		p.x = (float)dxy*kx;
		for (int ky=0; ky<nxy/2; ky++){		
			p.y = (float)dxy*ky;
			double val = box_in_circle(r,p)/(dxy*dxy);  //normalize to unity per voxel
			if(val >0.0 && opt.isset("cyltestpr"))printf("%2d %2d newval %9.5f\n",kx,ky,val);
			double dist = sqrt(p.x*p.x + p.y*p.y);
			if (dist <= F120_Rmin/sqrt(2.0)){
				vox[nxy*(my+ky)  +mx+kx]   = (float)val;
				vox[nxy*(my-ky-1)+mx+kx]   = (float)val;
				vox[nxy*(my+ky)  +mx-kx-1] = (float)val;
				vox[nxy*(my-ky-1)+mx-kx-1] = (float)val;
			}
		}
	}
	if (opt.isset("cylrange")){
		int j = opt.isset("cylrange");
		int z0 = opt.iopt(j+1);
		int z1 = max(1,z0);
		int z2 = opt.iopt(j+2);
		for (int z=z1; z<=z2; z++) for (int k=0; k<stride; k++) vox[z*stride+k] = vox[k];
		if (z0>0) for (int k=0; k<stride; k++) vox[k] = 0;
		printf("cyl z range limited to %d-%d\n",z0,z2);
	}
	else for (int z=1; z<nz; z++) for (int k=0; k<stride; k++) vox[z*stride+k] = vox[k];
	if (opt.isset("cyltest")){
		write_raw("cylvox.raw",vox,stride*nz);
		return 1;
	}
	return 0;
}

int const_fill(float *vox,AOptions &opt,int nxy,int nz,float val)
{
	int stride = nxy*nxy;
	for (int k=0; k<stride*nz; k++) vox[k] = 0.0f;

	for (int x=0; x<nxy; x++) for (int y=0; y<nxy; y++) if (roibox_cut(x,y)){
		for (int z=0; z<nz; z++) vox[stride*z+(y*nxy+x)]= val;
	}
	write_raw<float>("cfill_check.raw",vox,stride*nz);  // debug
	return 0;
}



int bigjob(SMfull &sm,AOptions &opt)
{
	int zstride = F120_NXYbins*F120_NXYbins;
	float *voxval = mycalloc<float>(zstride*F120_NZbins,"voxval");
	if (!voxval)return 1;	

	if (opt.isset("cylfill")) if(cyl_fill(voxval,opt,F120_NXYbins,F120_NZbins,F120_XYBin)) return 0;
	else if (opt.isset("ones")) const_fill(voxval,opt,F120_NXYbins,F120_NZbins,1.0f);
	else voxval[47*zstride+64*F120_NXYbins+64] =1.0f;  // TODO better phantoms needed!!

	// full size lor map here
	int stride = F120_NXY*F120_TrueNZ;
	float *tsum = NULL;
	
	if (opt.isset("tsum") || opt.isset("mapsum")){
		tsum = mycalloc<float>(stride*stride,"tsum/smap");
		if (!tsum) return 1;
	}

	// Compact lor map here.  NB Z size based on real detector not long version
	float *zdzmap = NULL;
	if (opt.isset("zdzmap")){
		zdzmap = mycalloc<float>(F120_DZstride*F120_DCstride,"zdzmap");
		if (!zdzmap) return 1;
	}


	//if (compute_forward(sm,voxval,tsum)) return 1;


	if (opt.isset("simple")){   // actually this is brocken
		int nx = opt.set_from_opt("nx",100);
		int ny = opt.set_from_opt("ny",70);
		int sector = opt.set_from_opt("sector",-1);
		if (simple_check(sm,opt,tsum,nx,ny,sector)) return 1;
	}

	else if (opt.isset("cuda")) {
		//LARGE_INTEGER CudaTime;
		//QueryPerformanceCounter(&CudaTime);
		cuda_do_forward_projection(sm,opt,voxval,tsum,zdzmap);
		//printf("Cuda time %.3f secs\n",timePassed(CudaTime));
	}

	else {
		LARGE_INTEGER ForwardTime;
		QueryPerformanceCounter(&ForwardTime);
		// do_forward_projection(sm,opt,voxval,tsum,zdzmap);
		host_do_forward_project(sm,voxval,zdzmap);
		printf("Host Forward time %.3f secs\n",timePassed(ForwardTime));
	}

	LARGE_INTEGER IOTime;
	QueryPerformanceCounter(&IOTime);

	if(opt.isset("voxval")) write_raw<float>("voxval.raw",voxval,zstride*F120_NZbins);
	if (opt.isset("tsum")) write_raw<float>("tsum.raw",tsum,stride*stride);
	if (opt.isset("zdzmap") && !opt.isset("cuda")) {
		write_raw<float>("host_small.raw",zdzmap,F120_DZstride*F120_DCstride);
		swizzle_buffer(zdzmap,145,288,1176,288,1,288*145);
		write_raw<float>("host_swizz.raw",zdzmap,F120_DZstride*F120_DCstride);
	}

	if (opt.isset("mapsum")){
		for (int k = 1; k < stride; k++) for (int j=0;j<stride;j++) tsum[j] += tsum[stride*k+j];
		write_raw<float>("mapsum.raw",tsum,stride);
	}

	//show_full_tsum(tsum);

	if(!opt.isset("cuda")) printf("IO time %.3f secs\n",timePassed(IOTime));

	if (zdzmap) free(zdzmap);
	if (tsum)   free(tsum);
	if (voxval)  free(voxval);


	return 0;
}

int make_dc(int c1,int c2)
{
	int dc = abs(c2-c1);
	if (c1 > c2) dc = F120_NXY-dc;  // fix logically negative dc values
	
	//if (dc < F120_DCmin || dc > F120_DCmax) return -1;  this check now done in cull program
	return dc-F120_DCmin;
}

template <typename T> int swizzle_buffer(T *a,int nz,int ny,int nx,int mz,int my,int mx)
{
	// reformat dim[n1,n2,n3] to [m1,m2,m3]  ( a permutation of original)
	int size = nz*ny*nx;
	T *b = (T *)malloc(size*sizeof(T));
	if (!b) return 1;
	for (int k=0; k<size; k++) b[k] = a[k];
	for (int z=0; z<nz; z++) for (int y = 0; y<ny; y++) for (int x=0; x<nx; x++){
		int k = (z*ny+y)*nx+x;
		int j =z*mz+y*my+x*mx;
		a[j] = b[k];
	}

	free(b);
	return 0;
}

int cuda_do_forward_projection(SMfull &sm,AOptions &opt,float *voxval,float *tsum,float *zdzmap)
{
	LARGE_INTEGER CudaTime;
	QueryPerformanceCounter(&CudaTime);
	cudaSM *dev_sm = NULL;
	cudaSM host_sm;
	VoxMap vm;
	cudaError_t cudaStatus;

	// sm for device
	if(setup_cuda_sm(sm,host_sm,&dev_sm)) return 1; 
	if(setup_cuda_vmap(vm)) return 1;  

	// sector constants to global device memory
	//int *map = vm.amap_x();   
	//cudaStatus = cudaMemcpyToSymbol(dev_map8_x,map,24*sizeof(int));
	//if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol map8_x failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	//map = vm.amap_y();
	//cudaStatus = cudaMemcpyToSymbol(dev_map8_y,map,24*sizeof(int));
	//if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol map8_y failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	//map = vm.amap_c();
	//cudaStatus = cudaMemcpyToSymbol(dev_map8_c,map,16*sizeof(int));
	//if (cudaStatus != cudaSuccess) { printf("cudaMemcpyToSymbol map8_c failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	// big buffers for device
	int zstride = F120_NXYbins*F120_NXYbins;
	float *dev_voxval = NULL;
	cudaStatus = cudaMalloc((void**)&dev_voxval,zstride*F120_NZbins*sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc dev_voxval failed [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	cudaStatus = cudaMemcpy(dev_voxval,voxval,zstride*F120_NZbins*sizeof(float),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to corners failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	float *dev_tsum = NULL;
	int stride = F120_NXY*F120_TrueNZ;
	cudaStatus = cudaMalloc((void**)&dev_tsum,stride*stride*sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc dev_tsum failed [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	cudaStatus = cudaMemset(dev_tsum,0,stride*stride*sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMemset to dev_tsum failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }

	float *dev_zdzmap = NULL;
	cudaStatus = cudaMalloc((void**)&dev_zdzmap,F120_DZstride*F120_DCstride*sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMalloc dev_zdzmap failed [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	cudaStatus = cudaMemset(dev_zdzmap,0,F120_DZstride*F120_DCstride*sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMemset to dev_zdzmap failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	cudaDeviceSynchronize();
	printf("Cuda setup time %.3f secs\n",timePassed(CudaTime));

	// efficiency sums (#voxels)  init from external file
	float *teffs = NULL;
	float *dev_teffs = NULL;
	if (make_buffers<float>(&teffs,&dev_teffs,zstride*F120_NZbins,"teffs")) return 1;
	if (read_buffers<float>(tvfile,zstride*F120_NZbins,teffs,dev_teffs,1.0f)) return 1;


	QueryPerformanceCounter(&CudaTime);
	printf("here we cuda go...\n");

	// first do forward projection
	for (int kv=0; kv<sm.voxels; kv++){
		//if (opt.isset("evenfaster")) {
		//	forward_project_even_faster<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,kv);
		//}
		if (opt.isset("faster")) {
		   // forward_project_faster<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,dev_teffs,kv,0);
		   // forward_project_faster<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,dev_teffs,kv,1);
		}
		else {
		   // forward_project<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,kv,0);
		   // forward_project<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,kv,1);
		}
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"forward_project kernel error: [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	}
	cudaDeviceSynchronize();
	printf("Cuda forward kernel time %.3f secs\n",timePassed(CudaTime));
	QueryPerformanceCounter(&CudaTime);

	// clear device buffer first!!!
	cudaStatus = cudaMemset(dev_voxval,0,zstride*F120_NZbins*sizeof(float));
	if (cudaStatus != cudaSuccess) { printf("cudaMemset to dev_voxval failed [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	// then do backward
	if (opt.isset("backproj")) for (int kv=0; kv<sm.voxels; kv++){
		if (opt.isset("bfast")) {
		   // backward_project_faster<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,dev_teffs,kv);
			//backward_project_faster<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,kv,1);
		}
		else{
			//backward_project<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,kv,0);
			//backward_project<<<64,64>>>(dev_sm,dev_voxval,dev_zdzmap,kv,1);
		}
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"back_project kernel error: [%s]\n",cudaGetErrorString(cudaStatus)); return 1; }
	}

	cudaDeviceSynchronize();
	printf("Cuda backward kernel time %.3f secs\n",timePassed(CudaTime));

	QueryPerformanceCounter(&CudaTime);
	cudaStatus = cudaMemcpy(zdzmap,dev_zdzmap,F120_DZstride*F120_DCstride*sizeof(float),cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to zdzmap failed: %s\n",cudaGetErrorString(cudaStatus));	return cudaStatus; }
	write_raw<float>("cuda_small.raw",zdzmap,F120_DZstride*F120_DCstride);
	//                     zo  yo  xo   z0->yn  yo->xn   xo->zn
	swizzle_buffer(zdzmap,145,288,1176,  288,     1,    288*145);
	write_raw<float>("cuda_swizz.raw",zdzmap,F120_DZstride*F120_DCstride);

	if (opt.isset("backproj")){
		cudaStatus = cudaMemcpy(voxval,dev_voxval,zstride*F120_NZbins*sizeof(float),cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to voxval failed: %s\n",cudaGetErrorString(cudaStatus));	return cudaStatus; }
		write_raw<float>("cuda_backp.raw",voxval,zstride*F120_NZbins);
	}
	cudaDeviceSynchronize();
	printf("Cuda IO end time time %.3f secs\n",timePassed(CudaTime));


	if (dev_zdzmap)   cudaFree(dev_zdzmap);
	if (dev_tsum)     cudaFree(dev_tsum);
	if (dev_voxval)   cudaFree(dev_voxval);
	if (host_sm.lors) cudaFree(host_sm.lors);
	if (host_sm.v)    cudaFree(host_sm.v);

	return 0;
}

// this version modeled on cuda kernel!
int host_do_forward_project(SMfull &sm,float *voxval,float *zdzmap)
{
	hex p;
	hex q;
	VoxMap vm;
	for (int kv =0; kv<sm.voxels; kv++){
		if(kv==sm.voxels-1) printf("kv %d\n",kv);
		else printf("kv %d\r",kv);
		int nlors = sm.v[kv].lors;
		p.x =    sm.v[kv].nx;
		p.y =    sm.v[kv].ny;
		//int vox_xypos =  p.y*F120_NXYbins+p.x;
		for (int kl=0; kl<16*nlors;kl++) for(int even=0;even<2;even++){
			int ids = kl%16;
			int idl = kl/16;
			uint key = sm.key(kv,idl);
			small_lor_from(key,p);
			float val = sm.val(kv,idl,even); 
			if (p.x==p.y) val *= 0.5f;     // fix for diagonal bug in sysmat  - present in all sectors thus counted twice in code below.
			if (val >0.0f){
				int dz = p.z2-p.z1;
				vm.hex_to_sector(ids%8,p,q);
				if (ids>7) mirror(q,F120_TrueNZ-1+even);
				int vox_xypos =  q.y*F120_NXYbins+q.x;

				//int zsm_offset = 0;
				int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum
				for (int sz=0; sz<F120_TrueNZ-dz; sz++){    // zloop (odd)
					int z1 = sz;
					int vzq = F120_TrueNZ-1+even - 2*(q.z1-z1);
					vzq = min(F120_NZbins-1,max(0,vzq));
					float qval = voxval[vzq*F120_NXYstride+vox_xypos];
					int dcq = make_dc(q.c1,q.c2);
					zdzmap[(q.c1+dcq*F120_NXY)*F120_DZstride+(sz+zsm_offset)] += val*qval;
				}
			}
		}
	}
	return 0;
}

int host_do_backward_projection(SMfull &sm,float *voxval,float *zdzmap)
{
	//
	//  Evaluate the Numerator sum: Sum[v] =  SM[t][v] T[t]  summing over lors t for each voxel v
	//
	// use 32 threads per lor each handles primary or mirror and even and odd
	//int id = threadIdx.x + blockIdx.x*blockDim.x;
	//int idt = id/32;   //  lor to process same for 8 threads per item
	//int ids = id%32;   //  which tread am i within 8 thread set: lor 8 sectors x 2 for proper & mirror
	//int even = (id%32)/16; // 0 or 1
	//int idstep = blockDim.x*gridDim.x/32;

	hex p;
	hex q;
	VoxMap vm;
	for (int kv =0; kv<sm.voxels; kv++){
		int nlors = sm.v[kv].lors;
		p.x = sm.v[kv].nx;
		p.y = sm.v[kv].ny;
		//if (ids==0 && kv == 0 )printf("idt %d nlors %d\n",idt,nlors);
		float div[2];
		div[0] = 1.0f/sm.v[kv].geff[0];
		div[1] = 1.0f/sm.v[kv].geff[1];

		//int vox_xypos =  p.y*F120_NXYbins+p.x;
		for (int kl=0; kl<16*nlors; kl++) for (int even=0; even<2; even++){
			int ids = kl%16;  // sector+ mirror sector
			int idl = kl/16;
			uint key = sm.key(kv,idl);
			lor_from(key,p);
			float val = sm.val(kv,idl,even);
			if (p.x==p.y) val *= 0.5f;

			//val *= 0.000001f;
			val *= div[even];
			if (val >0.0f){
				int dz = p.z2-p.z1;
				vm.hex_to_sector((ids/2)%8,p,q);
				if (ids%2) mirror(q,F120_TrueNZ-1+even);
				int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum

				int dcq = make_dc(q.c1,q.c2);
				int dcq_offset = (q.c1+dcq*F120_NXY)*F120_DZstride;
				//int dcq_offset = 10;
				//if (id <32 && kv==75) printf("%3d (%3d %3d %3d) %5d %5d\n",ids,q.c1,q.c2,dcq,dcq_offset,zsm_offset);
				int sz_max = F120_TrueNZ-dz;
				for (int sz=0; sz<sz_max; sz++){    // zloop (odd)
					int vzq = F120_TrueNZ-1+even - 2*(q.z1-sz);
					vzq = min(F120_NZbins-1,max(0,vzq));
					float tqval = zdzmap[dcq_offset+(sz+zsm_offset)];
					voxval[vzq*F120_NXYstride+q.y*F120_NXYbins+q.x] += tqval*val;				
				}				
			}
		}
	}

	return 0;
}

int map_chop(char *name_in, char *name_out)
{
	uint *map = mymalloc<uint>(F120_STride, "swizz in");   // single slice of full dataset
	uint *map_bit = map+F120_STride/4;

	FILE *fin = fopen(name_in, "rb");
	if (!fin) { printf("bad open for %s\n",name_in); return 1; }
	FILE *fout = fopen(name_out, "wb");
	if (!fout) { printf("bad open for %s\n",name_out); return 1; }
	int slice_in = 0;
	int slice_out = 0;
	for (int z1 = 0; z1<F120_NZ-24; z1++) for (int c1=0;c1<F120_NXY;c1++){
		if (fread(map,sizeof(uint),F120_STride,fin) != F120_STride) {printf("bad read for map slice %d\n",slice_in); return 1;}
		slice_in++;
		if (z1 >= 24) {
			fwrite(map_bit,sizeof(uint),F120_STride/2,fout); 
			slice_out++;
		}
	}
	fclose(fout);
	fclose(fin);
	printf("map %s chopped to %s s_in %d sout %d\n",name_in,name_out,slice_in,slice_out);
	return 0;
}



int map_swizz(char *name_in, char *name_out)
{
	uint *map =     mymalloc<uint>(F120_STride/2, "swizz in");   // single slice of full dataset
	float *zdzmap = mycalloc<float>(F120_DZstride*F120_DCstride,"swizz out ");
	if (!zdzmap) return 1;

	FILE *fin = fopen(name_in, "rb");
	if (!fin) { printf("bad open for %s\n",name_in); return 1; }
	int slice = 0;


	for (int z1 = 0; z1 < F120_TrueNZ; z1++){
		printf("z1 %2d slice %d\n",z1,slice);
		for (int c1 = 0; c1 < F120_NXY; c1++){
			if (fread(map,sizeof(uint),F120_STride/2,fin) != F120_STride/2) {printf("bad read for map slice %d\n",slice); return 1;}
			slice ++;				
			for (int z2=0;z2<F120_TrueNZ;z2++) 	for (int c2=0;c2<F120_NXY;c2++){
				float val = (float)map[z2*F120_NXY+c2];  
				if (val > 0.0f){
					quad p = {z1,c1,z2,c2};
					//p.z1  = z1;
					//p.c1 = c1;
					//p.z2 = z2;
					//p.c2 = c2;
					proper_lor(p);
					int dz = p.z2-p.z1;
					int zsm_offset = (dz*(97-dz))/2;   // this is ( 48+47+..) dz terms in sum  from cuda code!!
					int dcp = make_dc(p.c1,p.c2);
					//if(p.z1==5 && p.z2==15) printf("lor (%2d %3d)-(%2d %3d) val %8.1f dcp %3d dz %2d offset %4d\n",p.z1,p.c1,p.z2,p.c2,val,dcp,dz,zsm_offset);
					//if(dcp >= 0 && dcp <F120_DCsize) zdzmap[(p.c1+dcp*F120_NXY)*F120_DZstride+(p.z1+zsm_offset)] += val;
					if(dcp >= 0 && dcp <F120_DCsize) zdzmap[(p.z1+zsm_offset)*F120_DCstride  +(p.c1+dcp*F120_NXY)] += val;
				}
			}
		}			
	}
	fclose(fin);
	write_raw<float>(name_out,zdzmap,F120_DZstride*F120_DCstride);
	free(zdzmap);
	free(map);

	return 0;
}

int simple_check(SMfull &sm,AOptions &opt,float *tsum,int nx,int ny,int sector)
{
	// NB Z in [0,47] for crystals and [0,94] for voxels CARE

	printf("simple check for %d %d sector %d\n",nx,ny,sector);
	quad p;
	quad q;
	quad m; // mirror of p;

	VoxMap vm;
	//int bugs = 0;
	float sm_val0 = 0.0f;
	float sm_val1 = 0.0f;
	int stride = F120_NXY*F120_TrueNZ;

	int  dz_cut = opt.set_from_opt("dzcut",1);
	printf("dz_cut = %d\n",dz_cut);
	//return 1;

	for (int kv = 0; kv < sm.voxels; kv++){
		//if (sm.v[kv].nx != nx || sm.v[kv].ny != ny) continue;
		//printf("found voxel kv=%d lors = %d\n",kv,sm.v[kv].lors);
		printf("kv %d\r",kv);
		for (int kl = 0; kl < sm.v[kv].lors; kl++){
			uint key = sm.key(kv,kl);
			lor_from(key,p);

			//if (p.z1 != p.z2) continue; // debug!!!!
			if (abs(p.z1-p.z2) > dz_cut) continue; // debug!!!!

			sm_val0 = sm.val(kv,kl,0);
			sm_val1 = sm.val(kv,kl,1);	
			int dz = p.z2-p.z1;
			for (int s=0; s<8; s++){
				if (sector>=0 && s != sector) continue;  // sector = -1 does all
				vm.quad_to_sector(s,p,q);
				mirror(q,m,95);
				// TODO recover Z vertex necessary for real FP!!!!!  (=47/48-z1)
				//if (sm_val0> 0.0f) for (int vz=1; vz<F120_NZbins; vz+=2){  //oddds  TODO smart limits here
				//	
				//	int z1 = q.z1 + (vz - 95)/2;
				//	int z2 = q.z2 + (vz - 95)/2;
				//	int z3 = m.z1 + (vz - 95)/2;
				//	int z4 = m.z2 + (vz - 95)/2;
				
				if (sm_val0> 0.0f) for (int sz=0; sz<F120_TrueNZ-dz; sz++){ 				
					int z1 = sz;
					int z2 = sz+dz; 
					int z3 = sz;
					int z4 = sz+dz;

					//printf("lor %5d (%2d %3d)-(%2d %3d) -> %2d %2d val %9.5f\n",kl,p.z1,p.c1,p.z2,p.c2,z1,z2,sm_val);
					//if (z1>=0 && z2<F120_TrueNZ){
						tsum[(z1*F120_NXY+q.c1)*stride + z2*F120_NXY+q.c2] += sm_val0;
						tsum[(z2*F120_NXY+q.c2)*stride + z1*F120_NXY+q.c1] += sm_val0;
					//}
					//if (z3>=0 && z4<F120_TrueNZ){
						tsum[(z3*F120_NXY+m.c1)*stride + z4*F120_NXY+m.c2] += sm_val0;
						tsum[(z4*F120_NXY+m.c2)*stride + z3*F120_NXY+m.c1] += sm_val0;
					//}
				}

				// do evens?
				mirror(q,m,96);
				//if (sm_val1> 0.0f) for (int vz=0; vz<F120_NZbins; vz+=2){  //evens			
				//	int z1 = q.z1 + (vz - 96)/2;
				//	int z2 = q.z2 + (vz - 96)/2;
				//	int z3 = m.z1 + (vz - 96)/2;
				//	int z4 = m.z2 + (vz - 96)/2;
				if (sm_val1> 0.0f) for (int sz=0; sz<F120_TrueNZ-dz; sz++){  //evens			
					int z1 = sz;
					int z2 = sz+dz;
					int z3 = sz;
					int z4 = sz+dz;
					
					//printf("lor %5d (%2d %3d)-(%2d %3d) -> %2d %2d val %9.5f\n",kl,p.z1,p.c1,p.z2,p.c2,z1,z2,sm_val);
					//if (z1>=0 && z2 < F120_TrueNZ){
						tsum[(z1*F120_NXY+q.c1)*stride + z2*F120_NXY+q.c2] += sm_val1;
						tsum[(z2*F120_NXY+q.c2)*stride + z1*F120_NXY+q.c1] += sm_val1;
					//}
					//if (z3>=0 && z4 < F120_TrueNZ){
						tsum[(z3*F120_NXY+m.c1)*stride + z4*F120_NXY+m.c2] += sm_val1;
						tsum[(z4*F120_NXY+m.c2)*stride + z3*F120_NXY+m.c1] += sm_val1;
					//}
				}
			}       // end s  loop 
		}           // end kl loop
	}              // end kv loop
	
	return 0;
}

int show_full_tsum(float *tsum)
{
	int stride = F120_NXY*F120_TrueNZ;
	float *smap = mycalloc<float>(stride*stride,"tsum/smap"); 
	if (!smap) return 1;
	int zoffset = 0;
	for (int z1=0; z1<F120_TrueNZ; z1++)	{
		for (int z2=z1; z2<F120_TrueNZ; z2++){
			for (int c1=0; c1<F120_NXY; c1++) for (int dc=0; dc<F120_DCsize; dc++){
				int c2 = c1+dc+F120_DCmin;
				int dz = z2-z1;
				smap[(z1*F120_NXY+c1)*stride+(z2*F120_NXY+c2)] =tsum[((zoffset+dz)*F120_NXY+c1)*F120_DCsize+dc];
			}
		}
		zoffset += 48-z1;	
	}
	write_raw<float>("tsum_full.raw",smap,stride*stride);
	free(smap);
	return 0;
}

int compute_forward(SMfull &sm, float *voxval, float *tsum)
{
	// NB Z in [0,47] for crystals and [0,94] for voxels CARE
	quad p;
	quad m0; // odd mirror of p;
	quad m1; // even mirror of p;
	VoxMap vm;
	int bugs = 0;
	for (int kv = 0; kv < sm.voxels; kv++){
		if (bugs>0)printf("kv = %d\r",kv);
		int xv[8];
		int yv[8];
		int c1[8];
		int c2[8];
		// set voxel octet - good for all lors
		for (int s=0; s<8; s++) vm.xy_to_sector(s,xv[s],yv[s],sm.v[kv].nx,sm.v[kv].ny);
		// now loop over lors for this voxel octet
		if (bugs>0 && sm.v[kv].nx==64 && sm.v[kv].ny==64){
			printf("octet %d:",kv);
			for (int s=0; s<8; s++) printf(" (%d %d)",xv[s],yv[s]);
			printf("\n");
		}
		else printf("%d\r",kv);
		for (int kl = 0; kl < sm.v[kv].lors; kl++){
			//printf("kl=%d\n",kl);
			uint key = sm.key(kv,kl);
			lor_from(key,p);
			float sm_val0 =  sm.val(kv,kl,0);
			float sm_val1 =  sm.val(kv,kl,1);
			int dz = p.z2-p.z1;
			//if (p.c1 > p.c2) p.c2 += F120_NXY;
			int dc = abs(p.c2-p.c1);
			if (p.c1 > p.c2) dc = 288-dc;  // fix logically negative dc values
			if (dc < F120_DCmin || dc > F120_DCmax) continue;  // check now done in cull program
			dc -= F120_DCmin;

			int m0check = mirror(p,m0,95);  // posn in long detector needed here
			int m1check = mirror(p,m1,96);
			if (bugs>0){
				printf("kv/l %d %d p: (%2d %3d)-(%2d %3d) m0: (%2d %3d)-(%2d %3d) m1: (%2d %3d)-(%2d %3d) vals %8.5f %8.5f\n",kv,kl,p.z1,p.c1,p.z2,p.c2,m0.z1,m0.c1,m0.z2,m0.c2,m1.z1,m1.c1,m1.z2,m1.c2,sm_val0,sm_val1);
				bugs--;
			}
			int zoffset = 0;
			for (int s=0; s<8; s++){
				c1[s] = vm.c_to_sector(s,p.c1);
				c2[s] = vm.c_to_sector(s,p.c2);
			}
			int stride = F120_NXY*F120_TrueNZ;
			//swim each tube along z-axis of detector starting at z=0 and ending at 47-dz
			for (int zt=0; zt<F120_TrueNZ-dz; zt++){
				int p0_zv = 95 - 2*(p.z1-zt);  // zv generated at zvbin on 47/48 crystal boundry (voxel z=95)
				int p1_zv = 96 - 2*(p.z1-zt);  // zv generated at zvbin centre of crystal 48  (voxel z=96)
				int m0_zv = 95 - 2*(m0.z1-zt); // care bug fix 24/08/17 mirros keep primary voxel 
				int m1_zv = 96 - 2*(m1.z1-zt);  
				if(bugs>0){
					printf("zt=%2d raw p %2d %2d, p0 %2d p1 %2d m0 %2d m1 %2d offset %d\n",zt,p.z1,p.z2,p0_zv,p1_zv,m0_zv,m1_zv,zoffset);
					bugs--;
				}
				//if(zv0 < 0 || zv1 < 0) printf("zt=%d z1 %d z2 %d, zv0 %d zv1 %d\n",zt,p.z1,p.z2,zv0,zv1);
				
				for (int s=0; s<8; s++){
					if (p0_zv>=0) {
						tsum[(zt*F120_NXY+c1[s])*stride + (zt+dz)*F120_NXY+c2[s]] += sm_val0*voxval[(p0_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
						tsum[((zt+dz)*F120_NXY+c2[s])*stride + zt*F120_NXY+c1[s]] += sm_val0*voxval[(p0_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
					}
					if (m0_zv>=0) {
						tsum[(zt*F120_NXY+c1[s])*stride + (zt+dz)*F120_NXY+c2[s]] += sm_val0*voxval[(m0_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
						tsum[((zt+dz)*F120_NXY+c2[s])*stride + zt*F120_NXY+c1[s]] += sm_val0*voxval[(m0_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
					}
					//if(p0_zv>=0)  tsum[((zoffset+dz)*F120_NXY+p.c1 )*F120_DCsize+dc] += sm_val0*voxval[(p0_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
					//if(p1_zv>=0)  tsum[((zoffset+dz)*F120_NXY+p.c1 )*F120_DCsize+dc] += sm_val1*voxval[(p1_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
					//if(m0_zv>=0)  tsum[((zoffset+dz)*F120_NXY+m0.c1)*F120_DCsize+dc] += sm_val0*voxval[(m0_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
					//if(m1_zv>=0)  tsum[((zoffset+dz)*F120_NXY+m1.c1)*F120_DCsize+dc] += sm_val0*voxval[(m1_zv*F120_NZbins+yv[s])*F120_NXYbins+xv[s]];
				}
				zoffset += (F120_TrueNZ-zt);
			}
		}
	}
	printf("\n");
	return 0;
}

int roibox_cut(int x, int y)
{		
	
		double dx = ( abs(((double)x-63.5)) - 0.5 )*F120_XYBin;
		double dy = ( abs(((double)y-63.5)) - 0.5 )*F120_XYBin;  // corner closest to origin
		double limit = (double)F120_Rmin/sqrt(2.0);

		double dist = sqrt(dx*dx+dy*dy);
		//if (dist <= limit) return 0;

		return (dist <= limit) ? 1 : 0;
}

// this for both device and host
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

template <typename T> int read_buffers(char *name, int len, T *h_buf, T *d_buf, T rescale)
{
	if (read_raw<T>(name,h_buf,len)) return 1;
	if (rescale != (T)1.0) for (int k=0; k<len; k++) h_buf[k] *= rescale;
	cudaError_t cudaStatus = cudaMemcpy(d_buf,h_buf,len*sizeof(T),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy from file %s failed: [%s]",name,cudaGetErrorString(cudaStatus));  return 1; }
	//printf("read_buffers for %s\n",name);
	return 0;
}

template <typename T> int copy_buffer_to(int len, T *h_buf, T *d_buf)
{
	cudaError_t cudaStatus = cudaMemcpy(d_buf,h_buf,len*sizeof(T),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy from host buffer failed: [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	return 0;
}

template <typename T> int copy_buffer_from(int len, T *h_buf, T *d_buf)
{
	cudaError_t cudaStatus = cudaMemcpy(h_buf,d_buf,len*sizeof(T),cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr,"cudaMemcpy to host buffer failed: [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	return 0;
}


template <typename T> int clear_buffer(int len, T *d_buf)
{
	cudaError_t cudaStatus = cudaMemset(d_buf,0,len*sizeof(T));
	if (cudaStatus != cudaSuccess) { printf("cudaMemset to d_buf failed: [%s]",cudaGetErrorString(cudaStatus));  return 1; }
	return 0;
}

int cyl_buffer_fill_normalized(float *vox,double val)
{
	int stride = F120_NXYbins*F120_NXYbins;
	int nvox = stride*F120_NZbins;

	for (int k=0; k<nvox;k++) vox[k] = 0;

	int count = 0;
	for (int ky=0; ky<F120_NXYbins; ky++) for (int kx=0; kx<F120_NXYbins; kx++) if (roibox_cut(kx, ky)){
		for(int kz=0;kz<F120_NZbins;kz++) vox[kz*stride+(F120_NXYbins*ky+kx)] = (float)val;
		count += F120_NZbins;		
	}
	float nval = (float)val / (float)count;
	for (int k = 0; k<nvox; k++) vox[k] /= (float)(count);
	printf("buffer set to %.5e in ROI of %d voxels\n",nval,count);

	write_raw<float>("roi_start.raw",vox,nvox);

	return 0;
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