#ifndef ZDZMAP_H_
#define ZDZMAP_H_

#include "F120_long_defs.h"

// support for zdz format lor files.  S expected to be 48 or 96
// use simple functions for now - class syntax too difficult!??

template <int S> int zdz_offset(int dz) { 
	return (dz*(2*S+1-dz))/2; 
}

template <int S> int zdz_offset(int z1,int dz) { 
	return (z1 + (dz*(2*S+1-dz))/2); 
}

template <int S> int zdz_offset(int c1,int z1,int dz) { 
	int zstride = (S*(S+1))/2;
	return (c1*zstride + z1 + (dz*(2*S+1-dz))/2); 
}

template <int S> int zdz_offset(int dc,int c1,int z1,int dz) { 
	int zstride = (S*(S+1))/2;
	return ((dc*F120_NXY+ c1)*zstride + z1 + (dz*(2*S+1-dz))/2);
}

template <int S> int zdz_makedc(int c1,int c2){
	int dc = abs(c2-c1);
	if (c1 > c2) dc = F120_NXY-dc;  // fix logically negative dc values
	return dc-F120_DCmin;
}

// encapsulate compact z deltaZ lor map file. NB dc in [0,144], dz in [0,47] or [0,95]
//template <typename T,int S>  class ZDZmap {
//public:
//	T* zm;
//	int zstride;
//	int ZDZmap<T,S>::offset(int dz) { return (dz*(2*S+1-dz))/2; }  // NB S expected to be 48 or 96
//	int ZDZmap<T,S>::offset(int z1,int dz) { return (z1 + (dz*(2*S+1-dz))/2 ); }
//	int ZDZmap<T,S>::offset(int c1, int z1,int dz) { return (c1*zstride + z1 + (dz*(2*S+1-dz))/2 ); }
//	int ZDZmap<T,S>::offset(int dc,int c1, int z1,int dz) { return ((dc*F120NXY+ c1)*zstride + z1 + (dz*(2*S+1-dz))/2); }
//	int ZDZmap<T,S>::make_dc(int c1,int c2){
//		int dc = abs(c2-c1);
//		if (c1 > c2) dc = F120_NXY-dc;  // fix logically negative dc values
//		return dc-F120_DCmin;
//	}

//	ZDZmap() {
//		zstride = (S*(S+1))/2;
//		zm = NULL;
//		zm = (T *)calloc(F120_DCsize*F120_NXY*zstride, sizeof(T));
//		if (!zm)printf("calloc error in ZDZmap constuctor\n");
//	}
//	~ZDZmap() { if (zm) free(zm); }
//};


#endif
