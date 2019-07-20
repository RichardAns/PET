#ifndef SCANNER_H_
#define SCANNER_H_

char *F120_ID ="F120 Short";
// define F120 scanner
#define F120_Rmin 73.602000f
#define F120_Rmax 84.145916f
#define F120_Rsize 10.000000f
#define F120_Csize 1.592000f
#define F120_Cnum 12
#define F120_BPnum 24
#define F120_BZnum 4
#define F120_NXY 288
#define F120_NZ  48
#define F120_TrueNZ 48
#define F120_NDoi 16
#define F120_STride (F120_NXY*F120_NZ)
#define F120_BPhi 0.261799f
#define F120_BPface 19.104000f
#define F120_BZface 76.416000f
#define F120_TrueZface 76.416000f
//#define F120_BZface (F120_Csize*F120_NZ)

#define F120_BPfullface 19.379774f
#define F120_Thetacut 0.478823f
//#define F120_FCut1 74.219237f
//#define F120_FCut2 84.145916f
#define F120_LSOattn 0.087000f
//#define F120_LSOattn_Recip 0.087000f
#define F120_H2Oattn 0.009310f
//#define F120_XYBin 0.865759f  // this from header
// this for max square within inner radius
#define F120_XYBin 0.813194f
#define F120_ZBin 0.796f
#define F120_NXYbins 128
#define F120_NZbins 95
//#define F120_NZbins (F120_Cnum*F120_BZnum*2-1)

// these for sinograms
#define F120_NSegs ( (2*(F120_NZ)/3)+1 )
#define F120_SNX  128
#define F120_SNY  144
#define F120_SNZ  1567
#define F120_SNZ2  95
//#define F120_SNZ2  (F120_Cnum*F120_BZnum*2-1)

#endif