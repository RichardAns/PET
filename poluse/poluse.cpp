// poluse

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cx_host.h"  // host only cx
#include "scanner_host.h"
#include "timers.h"   // accurate timers for windows

struct cp_grid_map {
	float b[voxBox][voxBox];
	int x; // carteisian origin
	int y;
	int phi;  // polar voxel
	int r;
};

int main(int argc,char *argv[])
{
	if(argc < 2) {
		printf("usafe usepol <input zdzfile> <polmap> <output cartesian file>\n");
		return 0;	
	}

	int pol_size =  cryNum*detShortRings*radNum;  // NB order [ring, z, phi]
	int cart_size = voxNum*voxNum*detShortRings;  //          [z,    y,   x]
	int map_size =  cryNum*radNum;                //          [ring, phi]
	

	std::vector<float>        pol(pol_size);
	std::vector<float>       cart(cart_size);
	std::vector<cp_grid_map>  map(map_size);

	if(cx::read_raw(argv[1],pol.data(),pol_size)){printf("bad read on %s\n",argv[1]); return 1;}
	if(cx::read_raw(argv[2],map.data(),map_size)){printf("bad read on %s\n",argv[1]); return 1;}

	for(int r=0;r<radNum;r++) for(int z=0;z<detShortRings;z++) for(int p=0;p<cryNum;p++){
		float val = pol[(r*detShortRings+z)*cryNum+p];

		float vol_fact =  1.0f;  //2*r+1;
		int index = r*cryNum+p;
		if(val > 0.0f){
			int x0 = map[index].x;
			int y0 = map[index].y;
			for(int i=0;i<voxBox;i++) {
				int y = y0+i;
				if(y>=0 && y<voxNum) for(int j= 0;j<voxBox;j++){
					int x = x0+j;
					if(x>=0 && x <voxNum && map[index].b[i][j]>0.0f) cart[(z*voxNum+y)*voxNum+x] += vol_fact*val*map[index].b[i][j];
				}
			}
		}
	}

	cx::write_raw(argv[3],cart.data(),cart_size);

	return 0;
}


