#ifndef RECO_H_
#define RECO_H_

#include "lors.h"
#include "F120_long_defs.h"

// helpful stuff for cudareco
//
// Sector numberng scheme. Sector zero is master
//
//           \    |    /     (127,127)
//            \ 6 | 1 /
//             \  |  /
//          7   \ | /   0
//               \|/
//        --------+---------
//               /|\
//          4   / | \    3
//             /  |  \
//            / 5 | 2 \
//   (0,0)   /    |    \  

struct Sect2XY {
	int x;
	int y;
};
struct KVmap {

};

class VoxMap {
public:
	int map8_x[8][3];
	int map8_y[8][3];
	int map8_c[8][2];
	int zoffset[48];
	int nkv;
	int nxy;
	Sect2XY *p;
	int *XY2sect;

	int goodXY(int nx,int ny){
		double dx = (nx-64)*F120_XYBin;
		double dy = (ny-64)*F120_XYBin;  // corner closest to origin
		double limit = (double)F120_Rmin/sqrt(2.0);
		double dist = sqrt(dx*dx+dy*dy);
		if (dist <= limit) return 1;

		return 0;;
	}

	VoxMap() {
		int tmap8_x[8][3] ={ { 1,0,0 },{ 0,1,0 },{ 0,1,0 },   { 1,0,0 },   { -1,0,127 },{ 0,-1,127 },{ 0,-1,127 },{ -1,0,127 } };
		int tmap8_y[8][3] ={ { 0,1,0 },{ 1,0,0 },{ -1,0,127 },{ 0,-1,127 },{ 0,-1,127 },{ -1,0,127 },{ 1,0,0 },   { 0,1,0 } };
		//int tmap8_c[8][2] ={ { 0,1 },{ 72,-1 },{ 72,1 },{ 144,-1 },{ 144,1 },{ 216,-1 },{ 216,1 },{ 288,-1 } };
		//int tmap8_c[8][2] ={ { 0,1 },{ 72,-1 },{ 72,1 },{ 144,-1 },{ 144,1 },{ 216,-1 },{ 216,1 },{ 287,-1 } };
		int tmap8_c[8][2] ={ { 0,1 },{ 71,-1 },{ 72,1 },{ 143,-1 },{ 144,1 },{ 215,-1 },{ 216,1 },{ 287,-1 } };
		for (int i=0; i<8; i++)	for (int j=0; j<3; j++){
			map8_x[i][j] = tmap8_x[i][j];
			map8_y[i][j] = tmap8_y[i][j];
			if(j<2) map8_c[i][j] = tmap8_c[i][j];
		}
		zoffset[0] = 0;
		for (int k=0; k<47; k++) zoffset[k+1] = zoffset[k]+48-k;

		nkv = 0;
		for (int nx=64; nx<128; nx++) for (int ny = 64; ny<=nx; ny++) if (goodXY(nx,ny)) nkv++;
	
		int ngood = 0;
		p =(Sect2XY *)malloc(nkv*sizeof(Sect2XY));
		for (int nx=64; nx<128; nx++) for (int ny = 64; ny<=nx; ny++) if (goodXY(nx,ny)){
			p[ngood].x = nx;
			p[ngood].y = ny;
			ngood++;
		}
		if (ngood != nkv)printf("error ngood != nkv %d %d\n",ngood,nkv);
		else printf("%d good voxels in sector zero\n",nkv);

		XY2sect = (int *)malloc(128*128*sizeof(int));
		for (int j=0; j<128*128; j++)XY2sect[j] = -1;
		// assume origin in botton LH corner here
		for (int x=64; x<128; x++) for (int y=64; y<=x; y++) if (goodXY(x,y)) {
			Sect2XY p  ={ x,y };			
			Sect2XY q = { x,y };
			XY2sect[q.y*128+q.x] = 0;
			for (int s=1; s<8; s++){
				xy_to_sector(s,p,q);
				if (XY2sect[q.y*128+q.x] < 0 ) XY2sect[q.y*128+q.x] = s;
				//if (q.y==64 && q.x==0) printf("at (%3d %3d) from (%3d %3d)\n",q.x,q.y,p.x,p.y);
			}
		}
	}

	int *amap_x(void) { return &(map8_x[0][0]); }  // these to aid copy to cuda
	int *amap_y(void) { return &(map8_y[0][0]); }
	int *amap_c(void) { return &(map8_c[0][0]); }

	int c_to_sector(int s, int c)
	{
		int cs = map8_c[s][0]+map8_c[s][1]*c;
		if (cs<0)cs += 288;
		if (cs>=288)cs -= 288;
		return cs;
	}
	// this version overwires p
	int quad_to_sector(int s,quad &p)
	{
		p.c1 = map8_c[s][0]+map8_c[s][1]*p.c1;
		if (p.c1<0)p.c1 += 288;
		if (p.c1>=288)p.c1 -= 288;

		p.c2 = map8_c[s][0]+map8_c[s][1]*p.c2;
		if (p.c2<0)p.c2 += 288;
		if (p.c2>=288)p.c2 -= 288;

		if (p.z1==p.z2 && p.c1>p.c2){
			int t = p.c1;
			p.c1 = p.c2;
			p.c2 = t;
		}
		return 0;

	}
	
	// this version preserves p
	int quad_to_sector(int s,quad &p,quad &q)
	{
		q.c1 = map8_c[s][0]+map8_c[s][1]*p.c1;
		if (q.c1<0)q.c1 += 288;
		if (q.c1>=288)q.c1 -= 288;

		q.c2 = map8_c[s][0]+map8_c[s][1]*p.c2;
		if (q.c2<0)q.c2 += 288;
		if (q.c2>=288)q.c2 -= 288;
	
		q.z1 = p.z1;
		q.z2 = p.z2;
		if (q.z1==q.z2 && q.c1>q.c2){
			int tc= q.c1;
			q.c1 = q.c2;
			q.c2 = tc;
		}

		return 0;

	}

	int xy_to_sector(int s,Sect2XY &p,Sect2XY &q)
	{
		q.x = map8_x[s][0]*p.x + map8_x[s][1]*p.y + map8_x[s][2];
		q.y = map8_y[s][0]*p.x + map8_y[s][1]*p.y + map8_y[s][2];
		return 0;
	}

	int hex_to_sector(int s,hex &h)
	{
		int x = h.x;
		int y = h.y;
		h.x = map8_x[s][0]*x + map8_x[s][1]*y + map8_x[s][2];
	    h.y = map8_y[s][0]*x + map8_y[s][1]*y + map8_y[s][2];

		h.c1 = map8_c[s][0]+map8_c[s][1]*h.c1;
		if (h.c1<0)h.c1 += 288;
		if (h.c1>=288)h.c1 -= 288;

		h.c2 = map8_c[s][0]+map8_c[s][1]*h.c2;
		if (h.c2<0)h.c2 += 288;
		if (h.c2>=288)h.c2 -= 288;

		if (h.z1==h.z2 && h.c1>h.c2){
			int t = h.c1;
			h.c1 = h.c2;
			h.c2 = t;
		}
		return 0;
	}

	int hex_to_sector(int s,hex &p,hex &q)   //added 31/08/17
	{
		//int x = h.x;
		//int y = h.y;
		q.x = map8_x[s][0]*p.x + map8_x[s][1]*p.y + map8_x[s][2];
	    q.y = map8_y[s][0]*p.x + map8_y[s][1]*p.y + map8_y[s][2];

		q.c1 = map8_c[s][0]+map8_c[s][1]*p.c1;
		if (q.c1<0)q.c1 += 288;
		if (q.c1>=288)q.c1 -= 288;

		q.c2 = map8_c[s][0]+map8_c[s][1]*p.c2;
		if (q.c2<0)q.c2 += 288;
		if (q.c2>=288)q.c2 -= 288;

		q.z1 = p.z1;
		q.z2 = p.z2;
		if (q.z1==q.z2 && q.c1>q.c2){
			int t = q.c1;
			q.c1 = q.c2;
			q.c2 = t;
		}
		return 0;
	}

	int xy_to_sector(int s,int &x,int &y,int xin,int yin)
	{
		x = map8_x[s][0]*xin + map8_x[s][1]*yin + map8_x[s][2];
		y = map8_y[s][0]*xin + map8_y[s][1]*yin + map8_y[s][2];
		return 0;
	}

	int voxel_to_master(int &x,int &y)
	{
		int ix = x;
		int iy = y;
		int s = XY2sect[iy*128+ix];

		x = map8_x[s][2]+map8_x[s][0]*ix+map8_x[s][1]*iy;
		y = map8_x[s][2]+map8_x[s][0]*ix+map8_x[s][1]*iy;

		return 0;
	}

	int sector_from_voxel(int &x,int &y)
	{
		return XY2sect[y*128+x];
	}

	int show_sectors(void)
	{
		for (int y=127; y>=0; y--){
			printf("%3d| ",y);
			for (int x=0; x< 128; x++) {
				if (XY2sect[y*128+x]<0) printf("  ");
				else                    printf("%2d",XY2sect[y*128+x]);
			}
			printf(" | \n");
		}
		return 0;
	}

	float sector_sum(uint *vol, int x, int y)
	{
		float sum = 0.0;
		if (x <64) x = 127-x;
		if (y < 64 ) y = 127-y;

		if (x <64 || x > 127 || y <64 || y > 127) return 0.0f;
		int stride = F120_NXYbins*F120_NXYbins;
		for (int z = 0; z < F120_NZbins; z++){
			sum += vol[z*stride+(    y)*F120_NXYbins+(    x)];
			sum += vol[z*stride+(127-x)*F120_NXYbins+(    y)];
			sum += vol[z*stride+(127-y)*F120_NXYbins+(127-x)];
			sum += vol[z*stride+(    x)*F120_NXYbins+(127-y)];
			if (x == y) continue;  // on diagonal
			sum += vol[z*stride+(    x)*F120_NXYbins+(    y)];
			sum += vol[z*stride+(127-y)*F120_NXYbins+(    x)];
			sum += vol[z*stride+(127-x)*F120_NXYbins+(127-y)];
			sum += vol[z*stride+(    y)*F120_NXYbins+(127-x)];
		}
		if (sum >0.0f) printf("sector(%d,%d) sum = %f\n",x,y,sum);

		return sum;
	}
	
	~VoxMap() { 
		if (p) free(p);
		if (XY2sect) free(XY2sect);
	}
};

	//float map8_x[8][2] ={ { 1.0f,0.0f },{ 0.0f,1.0f },{ 0.0f,-1.0f },{ 1.0f,0.0f },{ -1.0f,0.0f },{ 0.0f,-1.0f },{ 0.0f,1.0f },{ -1.0f,0.0f } };
	//float map8_y[8][2] ={ { 0.0f,1.0f },{ 1.0f,0.0f },{ 1.0f,0.0f },{ 0.0f,-1.0f },{ 0.0f,-1.0f },{ -1.0f,0.0f },{ -1.0f,0.0f },{ 0.0f,1.0f } };
	//int   map8_c[8][2] ={ { 0,1 },{ 72,-1 },{ 72,1 },{ 144,-1 },{ 144,1 },{ 216,-1 },{ 216,1 },{ 288,-1 } };

#endif

