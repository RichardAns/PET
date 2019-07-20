#ifndef CUDAPET_H_
#define CUDAPET_H_

//#include "scanner.h"

// kermel configuration and big buffer size (product of these)
// SSlot increaded from 4 to 6 24/05/17 to allow for 2nd block hits

#define NThreads 256
#define NBlocks 160

#define MSlot 3
#define DSlot 1
#define DSize (NThreads*NBlocks*MSlot*DSlot)
#define SSlot 6
#define BSize (NThreads*NBlocks*MSlot*SSlot)
#define CSlot 2
#define CSize (NThreads*NBlocks*MSlot*CSlot)


typedef enum {CYLINDER,VOXEL,SPHERE,VTUBE,HALFVOX} SourceType;

class block {
public:
	float c[6];
	float lam[2];
	int face[2];
	int hits;
	block() { c[0] = -0.5f; c[1]=0.5f; c[2] = 4.0f; c[3]=5.0f;c[4] = 0.f;c[5] =1.0f; hits=0; };
	block(float x1,float x2,float y1,float y2,float z1,float z2) {c[0]=x1; c[1]=x2;c[2]=y1; c[3]=y2;c[4]=z1; c[5]=z2; hits=0; };
};

class ray {
public:
	float4 p;
	float4 n;
	float lam[4];
	int face[4];
	int hits;
	ray(){ hits=0; }
	ray(float4 pin,float4 nin) { p=pin; n=nin; hits=0; }
	float4 swim(float l) { return p+l*n; }  // NB helper_math.h provides lots of overloads for float4 etc.
};

struct Scanner {
	// define block based scanner
	float  Rmin;
	float  Rmax;
	float  Rsize;
	float  Csize;
	int    Cnum;
	int    BPnum;
	int    BZnum;
	int    NXY;
	int    NZ;
	int    NDoi;
	int    STride;
	float  BPhi;
	float  BPface;
	float  BZface;
	float  BPfullface;
	float  Thetacut;
	float  FCut1;
	float  FCut2;
	float  LSOattn;
	//float  LSOattn_Recip;
	float  H2Oattn;
	//#define F120_XYBin 0.865759f  // this from header
	// this for max square within inner radius
	float  XYBin;

	float  ZBin;
	int    NXYbins;
	int    NZbins;

	// these for sinograms
	int   SNX;
	int   SNY;
	int   SNZ;
	int   SNZ2;
};



class Source {
public:
	SourceType t;
	float4 o;
	float3 dim;  // (x,y,z) or (r,len,-) or (r,-,-)
	int3 nvox;
	Source(AOptions &opt,Scanner &scn);
	Source();
};

Source::Source(AOptions &opt,Scanner &scn){
	nvox = make_int3(0,0,0);
	o.w = 0.0f;
	if (opt.isset("genrot")) o.w = opt.set_from_opt("genrot",7.5f)*D2R;
	if (opt.isset("cyl")){
		t = CYLINDER;
		o.x = o.y = 0.0f;
		o.z = 0.5f*F120_BZface;
		dim.x = F120_Rmin/(float)sqrt(2.0);  // default to ROI
		dim.y = 0.5f*F120_BZface;  // half length
		dim.z = 0.0f; // unused
		if (opt.isset("cylpos"))  {int k=opt.isset("cylpos"); o.x += opt.fopt(k+1); o.y += opt.fopt(k+2); o.z += opt.fopt(k+3);}
		if (opt.isset("cylrad"))  dim.x = opt.fopt(opt.isset("cylrad")+1);
		if (opt.isset("cyllen"))  dim.y = opt.fopt(opt.isset("cyllen")+1);
		printf("source is cylinder centred at %.3f %.3f %.3f, having radius %.3f and total length %.3f rotation %.3f\n",o.x,o.y,o.z,dim.x,dim.y,o.w*R2D);
	}
	else if (opt.isset("sph")){
		t = SPHERE;
		o.x = o.y = 0.0f;
		o.z = 0.5f*F120_BZface;
		dim.x = 5.0f;
		dim.y = 0.0f; // unused
		dim.z = 0.0f; // unused
		if (opt.isset("sphpos"))  {int k=opt.isset("sphpos"); o.x += opt.fopt(k+1); o.y += opt.fopt(k+2); o.z += opt.fopt(k+3);}
		if (opt.isset("sphrad"))  dim.x = opt.fopt(opt.isset("sphrad")+1);
		printf("source is sphere centred at %.3f %.3f %.3f, having radius %.3f, rotation %.3f\n",o.x,o.y,o.z,dim.x,o.w*R2D);
		
	}
	else if (opt.isset("voxtube")){
		t = VTUBE;  //changed 13/11/17
		int k=opt.isset("voxtube");
		int xl = opt.iopt(k+1);
		int yl = opt.iopt(k+2);
		int xh = opt.iopt(k+3);
		int yh = opt.iopt(k+4);
		dim.x = F120_XYBin*(float)(xh-xl+1);
		dim.y = F120_XYBin*(float)(yh-yl+1);
		dim.z = F120_TrueZface;
		o.x = F120_XYBin*(float)(xl-F120_NXYbins/2)+0.5f*dim.x;
		o.y = F120_XYBin*(float)(yl-F120_NXYbins/2)+0.5f*dim.y;
		o.z = 0.5f*F120_BZface;   //use  48 zbins within 96 bin long detector
		nvox.x = xl;
		nvox.y = yl;
		nvox.z = F120_TrueNZ;
	}
	else if (opt.isset("vox")){
		
		if (opt.isset("halfvox")) t = HALFVOX;
		else t = VOXEL;
		if (opt.isset("highres")){
			o.x = o.y = F120_XYBin*0.25f;   // default voxel has x/y corner at system centre (128 is even)       
			dim.x = 0.5f*F120_XYBin;
			dim.y = 0.5f*F120_XYBin;
		}
		else{
			o.x = o.y = F120_XYBin*0.5f;   // default voxel has x/y corner at system centre (128 is even)       
			dim.x = F120_XYBin;
			dim.y = F120_XYBin;
		}
		dim.z = F120_ZBin; 
		o.z = 0.5f*F120_BZface;

		// NB check dim first to allow possible propagation to position changes. TODO allow voxel numbers to change
		if (opt.isset("voxdim"))  { int k=opt.isset("voxdim"); dim.x = opt.fopt(k+1); dim.y = opt.fopt(k+2); dim.z = opt.fopt(k+3); }
		if (opt.isset("voxnum"))  {
			int k=opt.isset("voxnum");
			int nx = opt.iopt(k+1);  if (nx >= 0 && nx <=F120_NXYbins) o.x = dim.x*((float)(nx-F120_NXYbins/2)+0.5f);
			int ny = opt.iopt(k+2);  if (ny >= 0 && ny <=F120_NXYbins) o.y = dim.y*((float)(ny-F120_NXYbins/2)+0.5f);
			int nz = opt.iopt(k+3);  if (nz >= 0 && nz <=F120_NZlongbins)  o.z = dim.z*((float)nz+1.0f);  // NB initial offset is 1/4 bin from edge, thus centre full bin width from edge.
			nvox.x = nx;
			nvox.y = ny;
			nvox.z = nz;
		}
		if (opt.isset("voxpos"))  { int k=opt.isset("voxpos"); o.x += opt.fopt(k+1); o.y += opt.fopt(k+2); o.z += opt.fopt(k+3); }
		
		if (t==VOXEL    )    printf("source is voxel centred at %.3f %.3f %.3f with sides %.3f %.3f %.3f Zlen is %10.6f bin is %8.6f, rotation %.3f\n",o.x,o.y,o.z,dim.x,dim.y,dim.z,F120_BZface,F120_ZBin,o.w*R2D);
		else if (t==HALFVOX) printf("source is half-voxel centred at %.3f %.3f %.3f with sides %.3f %.3f %.3f Zlen is %10.6f bin is %8.6f, rotation %.3f\n",o.x,o.y,o.z,dim.x,dim.y,dim.z,F120_BZface,F120_ZBin,o.w*R2D);
	}
	else{  // default big cylinder
		t = CYLINDER;
		o.x = o.y = 0.0f;
		o.z = 0.5f*F120_BZface;
		dim.x = 41.0f;
		dim.y = F120_BZface;
		dim.z = 0.0f; // unused
		printf("source is default cylinder at %.3f %.3f %.3f, radius %.3f and length %.3f, rotation %.3f\n",o.x,o.y,o.z,dim.x,dim.y,o.w*R2D);
	}
}


#endif
