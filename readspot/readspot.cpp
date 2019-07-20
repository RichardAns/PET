// readspot

#include <stdio.h>
#include <stdlib.h>
#include <random>
//#include "cx_host.h"
//#include "scanner_host.h"
#include "cx.h"
#include "scanner.h"

#include "timers.h"
#include "vector_types.h"
#include <vector>

bool hitsort_z1z2( const Hit &lhs,const Hit &rhs) { 

	// < means accending  i.e 00 comes before 01
	if(lhs.getkey() < rhs.getkey()) return true;
	return false;
}

bool hitsort_val( const Hit &lhs,const Hit &rhs) { 

	// < means accending  i.e 00 comes before 01
	if(lhs.val() < rhs.val()) return true;
	return false;
}

int main(int argc,char* argv[])
{
	if(argc <2){
		printf("usage readspot <spot file head> <sm file> <voxref1> <voxref2> wcut|0.1 zcut|64 dologs|0 verbose|0\n");
		return 0;
	}

	int voxref1 = 0;
	if(argc>3)voxref1 = atoi(argv[3]);  // radial ID1
	int voxref2 = voxref1;
	if(argc>4 && atoi(argv[4]) > voxref1 )voxref2 = atoi(argv[4]);  // radial ID2
	double wcut = 0.1;
	if(argc>5)wcut = atof(argv[5]);  // cut low fraction as cumulative percentage
	uint zcut = zNum-1;
	if(argc>6)zcut = atoi(argv[6]);  // max z2-z1 
	int dologs = 0;
	if(argc >7) dologs = atoi(argv[7]);
	int verbose = 0;
	if(argc >8) verbose = atoi(argv[8]);

	int slice        = spotNphi*spotNz;
	size_t bigslice  = zNum*cryNum;
	size_t size      = bigslice*slice;
	std::vector<uint> spot(size);
	FILE *slog = nullptr;
	FILE *mlog = nullptr;
	FILE *vlog = nullptr;
	FILE *wlog = nullptr;
	if(dologs){
		slog = fopen("slog.txt","w");
		mlog = fopen("mlog.txt","w");
		vlog = fopen("vlog.txt","w");
		wlog = fopen("wlog.txt","w");
	}

	// save hit numbers in seperate file <nfiles> <total hits> f0-hits ... fn_hits
	std::vector<uint> nhits(voxref2-voxref1+1+2); 
	nhits[0] = voxref2-voxref1+1;
	uint nhits_total = 0;
	char name[256]; 
	for(int vox = voxref1;vox <= voxref2;vox++){	
		sprintf(name,"%s%3.3d.raw",argv[1],vox);
		if(cx::read_raw(name,spot.data(),size,1) ) {printf("bad read on %s\n",name); return 1; }
		uint good = 0;
		uint bad = 0;
		size_t sl = 0;
		unsigned long long sum_sum = 0;
		unsigned long long sum_lors = 0;
		uint lmax_max = 0;
		std::vector<Hit> hits;
		unsigned long long sum_1 = 0;
		unsigned long long sum_10 = 0;
		unsigned long long sum_100 = 0;
		unsigned long long sum_1000 = 0;
		unsigned long long sum_10000 = 0;
		unsigned long long sum_100000 = 0;
		unsigned long long sum_1000000 = 0;
		for (int z1 = 0;z1<zNum;z1++) for(int c1=0;c1<cryNum;c1++){
			Lor l = {zNum-z1-1,c1,0,0};  // here z1 => zms1
			uint lors = 0;
			uint lmax = 0;
			uint lmin = 4000000000;
			int z0 =spot[sl];           spot[sl] = 0;
			int c0 = spot[sl+spotNphi]; spot[sl+spotNphi] = 0;
			unsigned long long slice_sum = 0;
			for(int z=0;z<spotNz;z++) for(int c=0;c<spotNphi;c++) {  // loop over spot image
				uint val = spot[sl+z*spotNphi+c];
				if(val < 1) continue;
				l.z2 = z+z0;
				l.c2 = (c+c0)%cryNum;  // NB might rotate 399->0
				// dz and dc cuts here
				if(abs(l.c1-l.c2) > cryDiffMax || abs(l.c1-l.c2) < cryDiffMin || (l.z1+l.z2) >= (int)zcut) continue; 
				slice_sum += val;
				lors++;
				lmax = std::max(lmax,val);
				lmin = std::min(lmin,val);
				Hit h(l,(float)val); // here we store hit
				hits.push_back(h);
				if(dologs)fprintf(slog,"  lor %2d (%2.2d-%3.3d) (%2.2d-%3.3d) %8.0f\n",lors,l.z1,l.c1,l.z2,l.c2,(double)val);
				if(val > 1000000)     sum_1000000++;
				else if(val > 100000) sum_100000++;
				else if(val > 10000)  sum_10000++;
				else if(val > 1000)   sum_1000++;
				else if(val > 100)    sum_100++;
				else if(val > 10)     sum_10++;
				else                  sum_1++;
			}
			sum_sum += slice_sum;
			sum_lors += lors;
			std::max(lmax_max,lmax);
			(slice_sum > 0) ? good++ : bad++;
			if(lors > 0 && dologs){
				fprintf(slog,"slice %2.2d-%3.3d hits %llu lors %d max %u min %u\n",z1,c1,slice_sum,lors,lmax,lmin);
				fprintf(mlog,"slice %2.2d-%3.3d hits %llu lors %d max %u min %u\n",z1,c1,slice_sum,lors,lmax,lmin);
			}
			sl += slice;
		}

		if(verbose)printf("grand sum %llu good slices %d bad %d sum_lors %llu maxlor %u\n",sum_sum,good,bad,sum_lors,lmax_max);
		if(verbose)printf("sums %llu %llu %llu %llu %llu %llu %llu\n",sum_1000000,sum_100000,sum_10000,sum_1000,sum_100,sum_10,sum_1);
		if(verbose)printf("Hits file has %llu lors\n",hits.size());
		size_t csize = hits.size();

		// this to drop all hits contibuting to lowest cumulative fraction of total hits

		long long clumsum = 0;
		double clumfrac = 0.0;
		long long clumcut = 0;
		float clumcut_val = 0.0f;
		std::sort<Hit *>(hits.data(),hits.data()+csize,hitsort_val);
		if(wcut > 0.0){
			for(size_t i=0;i<hits.size();i++){
				Lor l = hits[i].key_to();
				clumsum += (long long)hits[i].val();
				clumfrac = 100.0*(double)clumsum/(double)sum_sum;
				if(wcut > 0.0 && clumfrac >= wcut && clumcut==0) {clumcut = i; clumcut_val = hits[i].val();}
				if(dologs)fprintf(wlog,"slice %7lld (%2.2d-%3.3d) (%2.2d-%3.3d) %.0f %lld %.3f\n",i,l.z1,l.c1,l.z2,l.c2,hits[i].val(),clumsum,clumfrac);
			}
			
			size_t clumcut_size = hits.size()-clumcut;
			for(size_t j=0;j<clumcut_size;j++) hits[j] = hits[j+clumcut];
			hits.resize(clumcut_size);
			if(verbose)printf("wcut at %.0f reduced size from %lld to %llu lors\n",clumcut_val,csize,hits.size());
			csize = hits.size();
		}


		// this to resort in increasing z1/z2 order
		std::sort<Hit *>(hits.data(),hits.data()+csize,hitsort_z1z2);
		for(size_t i=0;i<hits.size();i++){
			Lor l = hits[i].key_to();
			if(dologs)fprintf(vlog,"lor %6lld (%2.2d-%3.3d) (%2.2d-%3.3d) %.0f\n",i,l.z1,l.c1,l.z2,l.c2,hits[i].val());
		}
		

		// this to save hits with 8-byte header
		//sprintf(name,"hits%3.3d.raw",vox);

		//cx::write_raw(name,&hsize,sizeof(long long));
		nhits[vox-voxref1+2] = (uint)hits.size();
		if(vox == voxref1) cx::write_raw(argv[2],hits.data(),hits.size());
		else              cx::append_raw(argv[2],hits.data(),hits.size());
		nhits_total += (uint)hits.size();
	}  // end vox loop

	nhits[1] = nhits_total;
	sprintf(name,"hits_total%3.3d-%3.3d.raw",voxref1,voxref2);
	cx::write_raw(name,nhits.data(),voxref2-voxref1+3);
	if(dologs){
		fclose(slog); 
		fclose(mlog);
		fclose(vlog);
		fclose(wlog);
	}

#if 1
	// this to check
	uint cksize = nhits[2];
	printf("check cksize = %d\n",cksize);
	std::vector<Hit> chits(cksize);
	//sprintf(name,"hits%3.3d.raw",voxref1);
	cx::read_raw(argv[2],chits.data(),cksize);
	Lor cl = chits[100].key_to();
	float v = chits[100].val();
	uint k = chits[100].getkey();
	printf("key %d %8.8x\n",100,k);
	printf("check lor %2d (%2.2d-%3.3d) (%2.2d-%3.3d) %8.0f\n",100, cl.z1, cl.c1, cl.z2, cl.c2, v);
#endif
	printf("size of sm_file %d rings %d records\n",nhits[0],nhits[1]);
	return 0;
}
