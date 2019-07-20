// smshow

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "cx.h"
#include "scanner.h"

#include "timers.h"
//#include "vector_types.h"
#include <vector>


int main(int argc,char *argv[])
{
	if(argc < 4){
		printf("Usage smform <sm_file> <sm_tab> <smfull output>\n");
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
	std::vector<sm_part>        sm(sm_size);
	std::vector<sm_full_part>   smfull(sm_size);
	if(cx::read_raw(argv[1],sm.data(),sm_size)) {printf("bad read for %s\n",argv[1]); return 1;}
	uint sp = 0;
	for(int ring =0; ring <radNum;ring++){
		for(uint k=smt[ring].sm_start;k<smt[ring].sm_end;k++){
			smfull[sp].key = sm[k].key;
			smfull[sp].val = sm[k].val;
			smfull[sp].key2 = ring;  // only need low 7-bits here rest reserved (maybe 10 bits)
			sp++;
		}
	}

	cx::write_raw(argv[3],smfull.data(),sm_size);
	cx::write_raw("smnew_size.raw",&sm_size,1);

	return 0;
}