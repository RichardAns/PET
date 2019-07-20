// smnorm

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "cx_host.h"
#include "scanner_host.h"

#include "timers.h"
//#include "vector_types.h"
#include <vector>

struct smtab {  // to complement monolithic sm table
	int ring;
	uint sm_start;
	uint sm_end;   // like iterator this is one past the end
	int  phi_steps;
};


int main(int argc,char *argv[])
{
	if(argc < 2){
		printf("Usage smnorm <sm_file> <hits file> <new sm> <new hits>.\n");
		return 0;	
	}

	std::vector<uint> nhits(radNum+2);
	if(cx::read_raw(argv[2],nhits.data(),radNum+2)){printf("bad read %s\n",argv[4]); return 1;}	
	if(nhits[0] != radNum){printf("bad nhits = %d, expected %d\n",nhits[0],radNum); return 1;}
	std::vector<smtab> smt(radNum);
	for (int k = 0; k < radNum; k++) {
		smt[k].phi_steps = 400;
		smt[k].ring = k;
		if(k==0){
			smt[0].sm_start = 0;
			smt[0].sm_end = nhits[2];
		}
		else {
			smt[k].sm_start = smt[k-1].sm_end; 
			if(k<radNum-1)smt[k].sm_end = smt[k].sm_start + nhits[2 + k];
			else smt[k].sm_end = nhits[1];
		}
	}
	for (int k = 0; k < radNum; k++) printf("smt[%2d] ring %3d steps %3d start %8u end %8u\n",k, smt[k].ring,smt[k].phi_steps,smt[k].sm_start,smt[k].sm_end);

	printf("sm size from hits[1] %8u\n",nhits[1]);

#if 0
	uint sm_size = nhits[1];
	std::vector<sm_part>   sm(sm_size);
	if (cx::read_raw(argv[1], sm.data(), sm_size)) { printf("bad read on sm_file %s\n", argv[1]); return 1; }
	for(int k=0;k<10;k++){
		Lor tl = key2lor(sm[k].key);
		float val = sm[k].val;
		printf("%8u (%2.3d %3.3d) (%2.2d %3.3d) %8.0f\n",k,tl.z1,tl.c1,tl.z2,tl.c2,val);
	}

	for(int r=0;r<radNum;r++){
		double sum = 0.0;
		for(uint k = smt[r].sm_start;k<smt[r].sm_end;k++) sum += sm[k].val;
		printf("ring %2d sum %10.0f\n",r,sum);
		double scale = 100.0/sum;
		for(uint k = smt[r].sm_start;k<smt[r].sm_end;k++) sm[k].val *= scale;
	}
	if (cx::write_raw(argv[3], sm.data(), sm_size)) { printf("bad write on sm file %s\n", argv[3]); return 1; }
#endif
	if (cx::write_raw(argv[4], smt.data(), radNum)) { printf("bad write on smtab file %s\n", argv[4]); return 1; }

	return 0;
}