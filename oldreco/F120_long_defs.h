#ifndef SCANNER_H_
#define SCANNER_H_

// these of for "long" F120 scanner having 8 rings instead of 4
// generating istotrpoic data in central voxels coverser full range
// of Z offsets in real scanner
char *F120_ID ="F120 Long";

// define long F120 scanner
#define F120_Rmin 73.602000f
#define F120_Rmax 84.145916f
#define F120_Rsize 10.000000f
#define F120_Csize 1.592000f
#define F120_Cnum 12
#define F120_BPnum 24
#define F120_BZnum 8
#define F120_NXY 288
#define F120_NZ 96
#define F120_TrueNZ 48
#define F120_SM_Zshift 24
#define F120_NDoi 16
#define F120_STride (F120_NXY*F120_NZ)
#define F120_BPhi 0.261799f
#define F120_BPface 19.104000f
#define F120_BZface  152.832f
#define F120_TrueZface 76.416000f
#define F120_BPfullface 19.379774f
#define F120_Thetacut 0.478823f
#define F120_LSOattn 0.087000f
#define F120_LSOattn_Recip 0.087000f
#define F120_H2Oattn 0.009310f

// these for  reco voxels
// this for max square within inner radius= Rmin/sqrt(2)
#define F120_XYBin 0.813194f
#define F120_ZBin 0.796f
#define F120_NXYbins 128
#define F120_NXYstride (F120_NXYbins*F120_NXYbins)
#define F120_NZlongbins 191
#define F120_NZbins 95
//#define F120_NZbins (F120_Cnum*F120_BZnum*2-1)

// these for sinograms (65 segments)
#define F120_NSegs ( (2*(F120_NZ)/3)+1 )
#define F120_SNX  128
#define F120_SNY  144
#define F120_SNZ  6207
#define F120_SNZ2  (F120_Cnum*F120_BZnum*2-1)

// 
// inclusive range of abs(c1-c2) values for accepted lors
#define F120_DCmin 72
#define F120_DCmax 216
#define F120_DCsize 145
#define F120_DCstride (F120_NXY*F120_DCsize)

// this is 48*49*(1/2)
#define F120_DZstride 1176
// this is 96*97*(1/2)
//#define F120_DZstride_big 4656

#define F120_KVnum 1661
#define F120_SZKstep (F120_NZbins*8) 
#define F120_SZKsize (F120_KVnum*F120_NZbins*8)

#endif
