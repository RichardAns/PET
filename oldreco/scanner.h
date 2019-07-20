#ifndef SCANNER_H_
#define SCANNER_H_


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

// define F120 scanner
#define F120_Rmin 73.602000f
#define F120_Rmax 84.145916f
#define F120_Rsize 10.000000f
#define F120_Csize 1.592000f
#define F120_Cnum 12
#define F120_BPnum 24
#define F120_BZnum 4
#define F120_NXY 288
#define F120_NZ 48
#define F120_NDoi 16
#define F120_STride (F120_NXY*F120_NZ)
#define F120_BPhi 0.261799f
#define F120_BPface 19.104000f
#define F120_BZface 76.416000f
#define F120_BPfullface 19.379774f
#define F120_Thetacut 0.478823f
#define F120_FCut1 74.219237f
#define F120_FCut2 84.145916f
#define F120_LSOattn 0.087000f
#define F120_LSOattn_Recip 0.087000f
#define F120_H2Oattn 0.009310f
//#define F120_XYBin 0.865759f  // this from header
// this for max square within inner radius
#define F120_XYBin 0.813194f

#define F120_ZBin 0.796f
#define F120_NXYbins 128
#define F120_NZbins 95

// these for sinograms
#define F120_SNX  128
#define F120_SNY  144
#define F120_SNZ  1567
#define F120_SNZ2  95


#ifndef PI
#define PI  3.141592654f 
#define PI2 6.283185307f 
#define R2D 57.29577951f 
#define D2R 0.017453292512f 
#endif

#endif