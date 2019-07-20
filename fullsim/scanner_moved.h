#pragma once
// this is host only version
#include "vector_types.h"

struct Lor {
	int z1;
	int c1;
	int z2;
	int c2;
};

// pack lor into 32-bit uint key, 
// | dz1 | C1 | dz2 | c2 |
// | 7   | 9  |  7  |  9 |
class Hit {
private:
	uint key;
	float v;
public:
	inline  void key_from(Lor &l) {
		key = 0;
		key = (l.z1<<25) | (l.c1<<16) | (l.z2<<9) | (l.c2);
	}
	inline Lor key_to(void) const {
		Lor l;
		l.c2 = key & 0x000001ff;
		l.z2 = (key>>9) & 0x0000007f;
		l.c1 = (key>>16) & 0x000001ff;
		l.z1 = (key>>25);
		return l;
	}
	inline  uint getkey(void) const { return key;}
	inline  float   val(void) const { return v;}
	Hit(){ key = 0; v = 0.0f; }  // necessary for std::vector allocation
	Hit(Lor &l,float val){
		key_from(l);
		v = val;
	}
};

__device__ void lor_from_key(uint &key, Lor &l)
{
	l.c2 = key & 0x000001ff;
	l.z2 = (key>>9) & 0x0000007f;
	l.c1 = (key>>16) & 0x000001ff;
	l.z1 = (key>>25);
}

__device__ void key_from_lor(uint &key, Lor &l)
{
	key = 0;
	key = (l.z1<<25) | (l.c1<<16) | (l.z2<<9) | (l.c2);
}


struct Ray {
	float3 a;
	float3 n;
	float lam1;
	float lam2;
};

struct Roi {
	float2 z;
	float2 r;
	float2 phi;
};

struct PRoi {   // Like Roi but add origin, for phantoms
	float2 z;   // z range
	float2 r;   // radial range
	float2 phi; // phi range
	float3 o;   // orign
};

/// Parameterise Scanner NB lengths in mm
constexpr int    cryNum     = 400;   // number of crystals in one ring
constexpr float  crySize    = 4.0f;  // size of square face
constexpr float  cryDepth   = 20.0f; // depth of crystal 
constexpr int    cryDiffMin = 100;   // min tranverse length of lor
constexpr int    cryDiffMax = 300;   // max transverse length of lor
constexpr int    cryDiffNum = (cryDiffMax-cryDiffMin+1);
constexpr float  cryStep    = (float)cryNum/cx::pi2<float>; // map phi to crystal
constexpr float  phiStep    = cx::pi2<float>/(float)cryNum; // map crystal to phi 

constexpr int    detRings  = 128;               // number of rings
constexpr int    detShortRings  = 64;               // number of rings
constexpr int    detMaxZDiff  = 63;               // max z difference
constexpr float  detRadius = 254.648f;         // radius of rings (detemined by cryNum and crySize)
constexpr float  detLen =  crySize*detRings; // axial length (max allowed z)
constexpr float  detShortLen = crySize*detShortRings; // axial length (max allowed z)
constexpr float  detStep =   1.0f/crySize;     // map z to ring
constexpr int    detZdZNum = detShortRings*(detShortRings+1)/2;  // Max z1/(z2-z1) combinations
constexpr int    cryCdCNum = cryNum*(cryDiffMax-cryDiffMin+1);   //  c1 / dc combinations

constexpr float  roiRadius = 200.0f;               // max radius for roi
constexpr int    radNum  = 100;                    // voxel transverse number
constexpr int    voxNum  = radNum*2;                    // voxel transverse number
constexpr float  voxSize  = 2.0f*roiRadius/voxNum; // voxel transverse size
constexpr float  voxStep  = 1.0f/voxSize;          // map tranverse distance to voxel

constexpr size_t mapSlice = cryNum*detRings/2;   // for sinogram maps enourmous
constexpr size_t mapShortSlice = cryNum*detShortRings;   // for sinogram maps enourmous
constexpr int    spotNphi = 24;                   // sinogram spot max phi size
constexpr int    spotNz = 24;                    // sinogram spot max z size

// end

