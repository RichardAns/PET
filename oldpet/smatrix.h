#ifndef SMATRIX_H_
#define SMATRIX_H_

#include <stdio.h>
//#include "mystuff.h"
#include "zdzmap.h"
#include "..\\reco\\reco.h"


#ifndef ushort
	typedef  unsigned char uchar;
	typedef  unsigned short ushort;
	typedef  unsigned int  uint;
#endif

struct smlor {
	uint key;
	ushort v[2];
};

bool smsort1(const smlor &lhs,const smlor &rhs) { return lhs.v[0] > rhs.v[0]; }    // > means decending
bool smsort2(const smlor &lhs,const smlor &rhs) { return lhs.v[1] > rhs.v[1]; }    // > means decending
bool smsort_key(const smlor &lhs,const smlor &rhs) { return lhs.key > rhs.key; }    // > means decending


struct SMlist {  // this for debug
	int z1;
	int c1;
	int z2;
	int c2;
	int dz;
	int dc;
	float val;
	int type;  // 0 => proper, 1 => mirror, 2=> self mirror, 3 = other( cflip??)
	int index;
};

bool slsort1(const SMlist &lhs,const SMlist &rhs) { 
	int lkey = (((lhs.z1+10)*512+lhs.c1)*64+lhs.z2)*512+lhs.c2;
	int rkey = (((rhs.z1+10)*512+rhs.c1)*64+rhs.z2)*512+rhs.c2;
	//printf("keys %d %d diff %d\n",lkey,rkey,lkey-rkey);

	//	lkey = (lkey<<9) & (lhs.c1);
//	lkey = (lkey<<6) & (lhs.z2);
//	lkey = (lkey<<9) & (lhs.c2);
//
//	int rkey = rhs.z1+10;
//	rkey = (rkey<<9) & (rhs.c1);
//	rkey = (rkey<<6) & (rhs.z2);
//
//	rkey = (rkey<<9) & (rhs.c2);


	return lkey < rkey;    

}    


struct SMfull_vox {
	int lor_offset;   // these do not need to be rebuilt after reading
	int lors;
	float scale;
	float geff[2];
	int nx;
	int ny;
};

class SMfull {
public:
	SMfull_vox *v;
	smlor *lors;
	int voxels;
	int numlors;

	SMfull() { voxels = numlors = 0; v = NULL; lors = NULL; }

	SMfull(int nvox,int nlors){
		voxels = nvox;
		numlors = nlors;
		v = (SMfull_vox *)malloc(sizeof(SMfull_vox)*voxels);
		if (!v) {
			printf("SMfull malloc error for v\n");
			voxels = 0;
			numlors = 0;
			return;
		}
		lors =(smlor *)malloc(sizeof(smlor)*numlors);
		if (!lors) {
			printf("SMfull malloc error for v\n");
			if (v) { free(v); v = NULL; }
			voxels = 0;
			numlors = 0;
			return;
		}
		return;
	}

	SMfull(char *name){
		FILE *fin = fopen(name,"rb");
		if (!fin) { 
			printf("SMfull constuctor error reading to %s\n",name); 
			voxels = 0;
			numlors = 0;
			v = NULL;
			lors = NULL;
			return; 
		}
		fread(&voxels,sizeof(int),1,fin);
		fread(&numlors,sizeof(int),1,fin);
		if (voxels > 0) {
			v = (SMfull_vox *)malloc(sizeof(SMfull_vox)*voxels);
			if (!v) {
				printf("SMfull malloc error for v\n");
				voxels = 0;
				numlors = 0;
				return;
			}
			fread(v,sizeof(SMfull_vox),voxels,fin);
		}
		if (numlors > 0) {
			lors =(smlor *)malloc(sizeof(smlor)*numlors);
			if (!lors) {
				printf("SMfull malloc error for v\n");
				if (v) { free(v); v = NULL; }
				voxels = 0;
				numlors = 0;
				return;
			}
			fread(lors,sizeof(smlor),numlors,fin);
		}
		fclose(fin);
		printf("SMfull read from file %s with %d voxels and %d lors\n",name,voxels,numlors);
		return;
	}

	// return rescaled value of lor[l] v[n]  // (no range check for speed)
	float val(int vox,int l,int n) { return (float)(lors[v[vox].lor_offset+l].v[n])*v[vox].scale; }
	uint  key(int vox,int l)       { return lors[v[vox].lor_offset+l].key; }

	int save(char *name){
		FILE *fout = fopen(name,"wb");
		if (!fout) { printf("SMfull error saving to %s\n",name); return 1; }
		fwrite(&voxels,sizeof(int),1,fout);
		fwrite(&numlors,sizeof(int),1,fout);
		if (voxels > 0 && v != NULL) fwrite(v,sizeof(SMfull_vox),voxels,fout);
		if (numlors > 0 && lors != NULL) fwrite(lors,sizeof(smlor),numlors,fout);
		fclose(fout);
		printf("SMfull saved to %s\n",name);
		return 0;
	}

	~SMfull() { if (v) free(v); if (lors) free(lors); }
};  // end SMfull

class SMbit {
public:
	double trys1[5];   //these for new gensys files
	double trys2[5];
	float scale;
	int lors;
	int nx;   // e.g. in [64,127]
	int ny;   // e.g. in [64,127]
	int nz;   // eg 95 to imply 95 and 96
	int ver;
	smlor *lor;

	SMbit() { lors = nx = ny =ver = 0; lor = NULL; }
	SMbit(float iscale,int ilors,int inx,int iny,int inz,int iver){
		scale = iscale;
		lors = ilors;
		nx = inx;
		ny = iny;
		nz = inz;
		ver = iver;
		lor = mycalloc<smlor>(lors,"list constuctor");
	}

	SMbit(char *name){
		FILE *fin= fopen(name,"rb");
		if (!fin) printf("bad open on %s for read\n",name); 
		if (fin){
			fread(&(trys1[0]),sizeof(double),5,fin);
			fread(&(trys2[0]),sizeof(double),5,fin);
			fread(&scale,sizeof(float),1,fin);
			fread(&lors,sizeof(int),1,fin);			
			fread(&nx,sizeof(int),1,fin);
			fread(&ny,sizeof(int),1,fin);
			fread(&nz,sizeof(int),1,fin);
			fread(&ver,sizeof(int),1,fin);

			lor = mycalloc<smlor>(lors,"file constuctor");
			if (lor)fread(lor,sizeof(smlor),lors,fin);
			fclose(fin);
			//printf("smbit read from %s\n",name);
		}
	}

	// return rescaled value of lor[l] v[n]  // (no range check for speed)
	float val(int l,int n) { return (float)(lor[l].v[n])*scale; }
	uint  key(int l) { return lor[l].key; }

	// this function intended to be used by default (empty) sm
	int peek(char *name){
		FILE *fin= fopen(name,"rb");
		if (!fin) {printf("bad open on %s for read\n",name); return 1;}

		fread(&(trys1[0]),sizeof(double),5,fin);
		fread(&(trys2[0]),sizeof(double),5,fin);
		fread(&scale,sizeof(float),1,fin);
		fread(&lors,sizeof(int),1,fin);			
		fread(&nx,sizeof(int),1,fin);
		fread(&ny,sizeof(int),1,fin);
		fread(&nz,sizeof(int),1,fin);
		fread(&ver,sizeof(int),1,fin);
		fclose(fin);

		return 0;
	}

	int re_read(char *name){
		if (lor) free(lor);
		lor = NULL;
		FILE *fin= fopen(name,"rb");
		if (!fin) { printf("bad open on %s for re_read\n",name); return 1; }
		
		fread(&(trys1[0]),sizeof(double),5,fin);
		fread(&(trys2[0]),sizeof(double),5,fin);
		fread(&scale,sizeof(float),1,fin);
		fread(&lors,sizeof(int),1,fin);
		fread(&nx,sizeof(int),1,fin);
		fread(&ny,sizeof(int),1,fin);
		fread(&nz,sizeof(int),1,fin);
		fread(&ver,sizeof(int),1,fin);
		
		lor = mycalloc<smlor>(lors,"file constuctor");
		if (lor)fread(lor,sizeof(smlor),lors,fin);
		else return 1;
		fclose(fin);
		//printf("smbit re_read from %s\n",name);

		return 0;
	}


	int show(void)	{
		int lors1 = (int)trys1[4];  // thisall broken 03/10/17 by new_gentrys
		int lors2 = (int)trys2[4];
		double ge1 = trys1[3] / trys1[2];
		double ge2 = trys2[3] / trys2[2];
		printf("voxel %3d %3d ver %d lors %d (%d & %d) geffs %7.5f %7.5f, rescale %7.2f\n",nx,ny,ver,lors,lors1,lors2,ge1,ge2,1.0/scale);
		//printf("t1:"); for (int k=0; k<5; k++) printf(" %.0f",trys1[k]); printf("\n");
		//printf("t2:"); for (int k=0; k<5; k++) printf(" %.0f",trys2[k]); printf("\n");
		return 0;
	}

	int show_lors(void)	{
		quad p;
		for (int k=0; k<lors; k++){
			lor_from(lor[k].key,p);
			float v0 = (float)lor[k].v[0]*scale;
			float v1 = (float)lor[k].v[1]*scale;
			printf("%5d (%2d %3d)-(%2d %3d) v0  %9.5f, v1 %9.5f\n",k,p.z1,p.c1,p.z2,p.c2,v0,v1);
		}
		return 0;
	}


	int get_trys(char *name,int zbin){
		FILE *fin = fopen(name, "rb");
		if (!fin) { printf("bad open for %s\n",name); return 1; }
		if (zbin==95) {
			fread(&(trys1[0]),sizeof(double),5,fin);
			printf("%s :trys1 %.0f %.0f %.0f %.0f %.0f\n",name,trys1[0],trys1[1],trys1[2],trys1[3],trys1[4]);
		}
		else if (zbin==96) {
			fread(&(trys2[0]), sizeof(double),5,fin);
			printf("%s :trys2 %.0f %.0f %.0f %.0f %.0f\n",name,trys2[0],trys2[1],trys2[2],trys2[3],trys2[4]);
		}
		else { printf("error in get_trys, expected either 95 or 96 for zbin=%d, file name %s\n",zbin,name); fclose(fin); return 1; }
		fclose(fin);
		return 0;
	}

	int get_new_trys(int xpos,int ypos,int zpos){
		char name[256];
		double newtry[10];
		sprintf(name,"gentrys\\new_gentrys_%d_%d_%d.raw",xpos,ypos,zpos);
		read_raw_quiet<double>(name,&(newtry[0]),10);
		//FILE *fin = fopen(name,"rb");
		//if (!fin) { printf("bad open for %s\n",name); return 1; }
		if (zpos==95) {
			//fread(&(newtry[0]),sizeof(double),10,fin);
			trys1[0] = newtry[0]; // k2 good
			trys1[1] = newtry[8]; // was k2 bad - now new overall efficency in short detector (about 44%)
			trys1[2] = newtry[2]; // k2 all gen (bad = all-good, if still required)
			trys1[3] = newtry[3]; // k3 really good
			trys1[4] = newtry[4]; // number of lors
			printf("%s :trys1 %.0f %.3f %.0f %.0f %.0f\n",name,trys1[0],trys1[1]*100.0,trys1[2],trys1[3],trys1[4]);
		}
		else if (zpos==96) {
			//fread(&(newtry[0]),sizeof(double),10,fin);
			trys2[0] = newtry[0]; // k2 good
			trys2[1] = newtry[8]; // was k2 bad - now new overall efficency in short detector (about 44%)
			trys2[2] = newtry[2]; // k2 all gen
			trys2[3] = newtry[3]; // k3 really good
			trys2[4] = newtry[4]; // number of lors
			printf("%s :trys2 %.0f %.3f %.0f %.0f %.0f\n",name,trys2[0],trys2[1]*100.0,trys2[2],trys2[3],trys2[4]);
		}
		else { printf("error in get_trys, expected either 95 or 96 for zbin=%d, file name %s\n",zpos,name); return 1; }
		//fclose(fin);
		return 0;
	}

	int save(char *name){
		FILE *fout = fopen(name,"wb");
		if (!fout) { printf("bad open on %s for write\n",name); return 1; }
		fwrite(&(trys1[0]),sizeof(double),5,fout);
		fwrite(&(trys2[0]),sizeof(double),5,fout);
		fwrite(&scale,sizeof(float),1,fout);
		fwrite(&lors,sizeof(int),1,fout);
		fwrite(&nx,sizeof(int),1,fout);
		fwrite(&ny,sizeof(int),1,fout);
		fwrite(&nz,sizeof(int),1,fout);
		fwrite(&ver,sizeof(int),1,fout);

		fwrite(lor,sizeof(smlor),lors,fout);
		fclose(fout);
		//printf("file %s written\n",name);
		return 0;
	}
	~SMbit(){ if (lor) free(lor); }
}; // end SMbit

//class FullSM {
//public:
//	int nsm;         // number of smbits in full sm
//	float *geff;     // mapsum/ntrys     (size nsm*2)
//	float *norm;     // 1,000,000/mapsum (size nsm*2)
//	uint nsm_lors;   // total number of lors summed over  all sm  
//	smlor *sm;       // all lors smlors (size big)
//	uint *sm_start;  // start point in sm for given voxel
//	uint *sm_end;    // end point in sm for given voxel
//};
#endif
