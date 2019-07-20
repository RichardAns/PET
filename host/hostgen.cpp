// hostgen  host calucalation of PI
#include <stdio.h>
#include <stdlib.h>
#include <random>

#include "cx_host.h"  // host only cx
#include "timers.h"   // accurate timers for windows

struct float3 {
	float x;
	float y;
	float z;
};

//cyl2d origin len rad nz nphi zstep phistep
struct cyl2D {
	float3 o; // origin
	float len;  // length
	float r;  // radius
	int nz;   // len bins
	int nphi; // phi bins on surface
	float zstep;
	float phistep;
};
//cyl3d origin len rad nz nxy zstep xystep
struct cyl3D {
	float3 o;  // origin
	float len; // length
	float r;   // radius
	int nz;    // z bins
	int nxy;   // x-y bins
	float zstep;
	float xystep;
};

struct Lor {
	int z1;
	int p1;
	int z2;
	int p2;
};

struct Ray {
	float3 a;
	float3 n;
	float lam1;
	float lam2;
};

int ray_to_cyl(Ray &g,Lor &l,cyl2D c)
{
	//swim to cyl: slove quadratic
	float A = g.n.x*g.n.x + g.n.y*g.n.y;
	float B = g.a.x*g.n.x + g.a.y*g.n.y;  // factors of 2 ommited as they cancel
	float C = g.a.x*g.a.x + g.a.y*g.a.y - c.r*c.r;
	float D = B*B-A*C;
	float rad = sqrtf(D);
	g.lam1 = (-B+rad)/A;  // gamma1
	float z1 = g.a.z+g.lam1*g.n.z;
	g.lam2 = (-B-rad)/A;  // gamma2
	float z2 = g.a.z+g.lam2*g.n.z;  
							  
	if(z1 >= 0.0f && z1 < c.len && z2 >= 0.0f && z2 < c.len){
		float x1  = g.a.x+g.lam1*g.n.x;
		float y1  = g.a.y+g.lam1*g.n.y;
		float phi = atan2(y1,x1)+cx::pi<float>;
		l.z1 = std::min(c.nz-1,  (int)(z1*c.zstep)   );
		l.p1 = std::min(c.nphi-1,(int)(phi*c.phistep));
		float x2  = g.a.x+g.lam2*g.n.x;
		float y2  = g.a.y+g.lam2*g.n.y;
		phi = atan2(y2,x2)+cx::pi<float>;
		l.z2 = std::min(c.nz-1,  (int)(z2*c.zstep)   );
		l.p2 = std::min(c.nphi-1,(int)(phi*c.phistep));
		return 1;
	}

	return 0;
}

int main(int argc, char *argv[])
{
	printf("f pi is %.10f\n",cx::pi<float>);
	printf("d pi is %.10f\n",cx::pi<double>);

	if(argc < 2){
		printf("usage simple threads block ngen seed R L r l\n");
		return 0;
	}
	//cyl2d origin len rad nz nphi zstep phistep
	cyl2D det_hit = {0.0f,0.0f,0.0f, 200.0f,400.0f ,256,256, 256.0f/(200.0f), 256.0f/cx::pi2<float> };  // outer cylinder	

	printf("cyl2d org (%.1f %.1f %.1f)  zlen %.1f radius %.1f nz %d nphi %d zstep %.3f phistep %.3f\n",
		          det_hit.o.x,det_hit.o.y,det_hit.o.z,det_hit.len,det_hit.r,det_hit.nz,det_hit.nphi,det_hit.zstep,det_hit.phistep);

	//cyl3d origin len rad nz nxy zstep xystep
	cyl3D roi =  {0.0f,0.0f,0.0f, 200.0f,200.0f ,256,256, 256.0f/(200.0f), 256.0f/(2.0f*200.0f)};  // inner cylinder
	printf("cyl2d org (%.1f %.1f %.1f)  zlen %.1f radius %.1f nz %d nxy %d zstep %.3f xystep %.3f\n",
	                roi.o.x,roi.o.y,roi.o.z,roi.len,roi.r,roi.nz,roi.nxy,roi.zstep,roi.xystep);

	uint threads = 256; if(argc>1) threads = atoi(argv[1]);
	uint blocks = 1024; if(argc>2) blocks  = atoi(argv[2]);

	uint size = blocks*threads;
	long long  ngen = atoll(argv[3])*1000000ll;
	uint tries = (ngen+size-1)/size;
	ngen = (long long)tries*(long long)size;

	std::random_device rd;
	long long seed = rd(); if (argc > 4) seed = atoi(argv[4]);

	std::default_random_engine gen(seed);
	std::uniform_int_distribution<int>  idist(0, 2147483647); // maps gen to [0,2^31) as ints
	double idist_scale = 1.0/2147483647.0;

	std::vector<uint> hits(det_hit.nz*det_hit.nphi);
	std::vector<uint> roi_points(roi.nz*roi.nxy*roi.nxy);

	cx::MYTimer tim;
	Ray g;
	Lor lor;
	double good = 0.0;
	for(uint i=0;i<size;i++) for(uint j=0;j<tries;j++){
		float phi = cx::pi2<float>*idist_scale*idist(gen);
		float r = sqrtf(idist_scale*idist(gen))*roi.r;
		g.a.x = r*cosf(phi);
		g.a.y = r*sinf(phi);
		g.a.z = roi.len*idist_scale*idist(gen);

		// generate isotropic back to back gammas
		phi = cx::pi2<float>*idist_scale*idist(gen);;
		float theta = acosf(1.0f-2.0f*idist_scale*idist(gen));
		g.n.x = sinf(phi)*sinf(theta);
		g.n.y = cosf(phi)*sinf(theta);
		g.n.z = cosf(theta);
		if(ray_to_cyl(g,lor,det_hit)){
			good++;
			float x1  = g.a.x+g.lam1*g.n.x;
			float y1  = g.a.y+g.lam1*g.n.y;
			float z1  = g.a.z+g.lam1*g.n.z;
			phi = atan2f(y1,x1)+cx::pi<float>;
			int az = std::min(det_hit.nz-1,  (int)(z1*det_hit.zstep)   );
			int ay = std::min(det_hit.nphi-1,(int)(phi*det_hit.phistep));
			if(az < 0 || az >= det_hit.nz || ay < 0 || ay >= det_hit.nphi) printf("i %d j %dL1 %d %d\n",i,j,az,ay);
			else hits[ay*det_hit.nz+az] += 1;
			float x2  = g.a.x+g.lam2*g.n.x;
			float y2  = g.a.y+g.lam2*g.n.y;
			float z2  = g.a.z+g.lam2*g.n.z;
			phi = atan2f(y2,x2)+cx::pi<float>;
			az = std::min(det_hit.nz-1,  (int)(z2*det_hit.zstep)   );
			ay = std::min(det_hit.nphi-1,(int)(phi*det_hit.phistep));
			if(az < 0 || az >= det_hit.nz || ay < 0 || ay >= det_hit.nphi) printf("i %d j %d L2 %d %d\n",i,j,az,ay);
			else hits[ay*det_hit.nz+az] += 1;

			int ax = (int)((g.a.x+roi.r)*roi.xystep);  // shift origin for plots
			ay = (int)((g.a.y+roi.r)*roi.xystep);
			az = (int)(g.a.z*roi.zstep);
			az = std::min(roi.nz-1,az);
			//if(id==0) printf("r %f phi %f a.x %f a.y %f a.z %f voxel %d %d %d\n",r,phi,a.x,a.y,a.z,ax,ay,az);
			if(ax <0 || ax >= roi.nxy || ay < 0 || ay >= roi.nxy || az < 0 || az >= roi.nz) printf("bad roi %d %d %d\n",ax,ay,az); 
			else roi_points[(az*roi.nxy+ay)*roi.nxy+ax] += 1; 

		}
	}

	tim.add();
	
	double eff = 100.0*good/(double)ngen;
	printf("ngen %lld good %.0f eff %.3f%% time %.3f ms\n",ngen,good,eff,tim.time());

	cx::write_raw("roi_host.raw",roi_points.data(),roi.nz*roi.nxy*roi.nxy);
	cx::write_raw("hits_host.raw",hits.data(),det_hit.nz*det_hit.nphi);

	return 0;
}
