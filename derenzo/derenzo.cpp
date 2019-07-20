// dernezo

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include "cx.h"

struct point {
	double x;
	double y;
};

struct dpoint {
	double x;
	double y;
	double r;
};

int hex_mesh(std::vector<dpoint> &p, double r,double theta,double cut)
{
	// Derenzo Phantom has spots spaced by 2 x diameter.
	// arranged in 6 hexagonal segments
	double s = 0.5;
	double c = sqrt(3)/2.0;
	double vscale = c;    

	point m = {-s,c }; // left edge of triangle
	double st = sin(theta);
	double ct = cos(theta);

	double dist = 2.0*r;
	double dist2 = 0.0;

	dpoint spot;
	double sx = 0.0;
	double sy = 1.6*dist;  // dit from centre of first spot
	spot.x =  sx*ct + sy*st;
	spot.y = -sx*st + sy*ct;
	spot.r = r;
	p.push_back(spot);

	int nspots = 2;
	
	while(1){
		double dist_edge = 2.0*dist*(double)(nspots-1);
		sx = m.x*dist_edge;
		sy = m.y*dist_edge + 1.6*dist;
		dist2 = sqrt(sx*sx+sy+sy);
		if(dist2 >= cut) break;
		for(int k=0;k<nspots;k++){
			spot.x =  sx*ct + sy*st;
			spot.y = -sx*st + sy*ct;
			p.push_back(spot);
			sx += 2.0*dist;
		}
		nspots++;		
		//printf("%d %f %f\n",nspots,dist2,cut);
	} 
	return p.size();
}

int main(int argc,char *argv[])
{
	const char *torun = "D:\\all_code\\vs17\\pet\\x64\\Release\\fullsim.exe";
	if(argc < 8){
		printf("usage: derenzo r1 r2 .. r6 rcut\n");
		return 0;
	}
	FILE *flog = fopen("dergen.bat","w");
	double theta = 0.0;
	std::vector<dpoint> p;
	for (int k=0;k<6;k++) {
		hex_mesh(p,atof(argv[k+1]),theta,atof(argv[7]));
		theta += cx::pi<double>/3.0;
	}
	printf("%d point in phantom\n",(int)p.size());
	for(int k=0;k<p.size();k++){
		printf("%3d %12.6f %12.6f %6.1f\n",k,p[k].x,p[k].y,p[k].r);
		fprintf(flog,"%s 2 256 1024 %d %d derenzo_full.raw 0 %6.2f 0 360 64 192 %12.6f %12.6f 0 0 1\n",
			torun, (int)(10.0*p[k].r*p[k].r), 1234+k*37, p[k].r, p[k].x, p[k].y);
	}
	return 0;
}