// smshow

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
	if(argc < 4){
		printf("Usage smshow <sm_file> <sm_tab> <ring> [nhist file]\n");
		return 0;	
	}

	if(argc >4){
		std::vector<uint> nhits(radNum+2);
		if(cx::read_raw(argv[4],nhits.data(),radNum+2)){printf("bad read %s\n",argv[4]); return 1;}	
		if(nhits[0] != radNum){printf("bad nhits = %d, expected %d\n",nhits[0],radNum); return 1;}
		printf("Nhits[0-1] %u %u\n",nhits[0],nhits[1]);
		uint start = 0;
		uint end = 0;
		uint len = 0;
		for(int k=0;k<radNum;k++) {
			end += nhits[k+2];
			len = nhits[k+2];
			printf("ring %3d start %8u end %8u len %8u\n",k,start,end, len);
			start += len;
		}
		return 0;
	}

	std::vector<smtab> smt(radNum);
	if(cx::read_raw(argv[2],smt.data(),radNum)) {printf("bad read for %s\n",argv[2]); return 1;}

	uint sm_size = 0;
	for (int k = 0; k < radNum; k++) {
		printf("smt[%2d] ring %3d steps %3d start %8u end %8u\n",k, smt[k].ring,smt[k].phi_steps,smt[k].sm_start,smt[k].sm_end);
		sm_size += smt[k].sm_end - smt[k].sm_start;
	}
	printf("sm size from sum %8u\n",sm_size);

	if(sm_size != smt[radNum-1].sm_end){printf("size mismatch error\n"); return 1;}
	std::vector<sm_part>   sm(sm_size);
	if(cx::read_raw(argv[1],sm.data(),sm_size)) {printf("bad read for %s\n",argv[1]); return 1;}

	int ring = atoi(argv[3]);
	if(ring <0 || ring >=radNum) return 0;

	char name[256];
	sprintf(name,"smlist_r%3.3d.txt",ring);
	FILE *flog = fopen(name,"w");
	fprintf(flog,"Ring %3d data\n",ring);
	for(uint k=smt[ring].sm_start;k<smt[ring].sm_end;k++){
		Lor tl = key2lor(sm[k].key);
		float val = sm[k].val;
		fprintf(flog,"%8u (%2.3d %3.3d) (%2.2d %3.3d) %8.0f\n",k,tl.z1,tl.c1,tl.z2,tl.c2,val);
	}

	fclose(flog);
	printf("file %s written\n",name);

	return 0;
}