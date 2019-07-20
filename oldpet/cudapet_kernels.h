#ifndef CUDAPET_KERNELS_H_
#define CUDAPET_KERNELS_H_

//-------  introduce some device global variables to save arguments --------------
__device__ uint4 *devglob_gentrys;   // devglob.x count for all  attempts, globgen.y counts for for good events
__device__ __constant__ Scanner dev_scn;
//--------------------------------------------------------------------------------


__host__ __device__  bool check_range(float value,float min,float max) {
	return (value >= min) && (value <= max);
}

//class __device_builtin__ __builtin_align__(16) float4p{
//public:
//	float x, y, z, w;
//	inline __host__ __device__ float4p() { x=y=z=w=0.0f; };
//	inline __host__ __device__ float4p(float s) { x=y=z=w=s; };
//	inline __host__ __device__ float4p(float a,float b,float c) { x=a; y=b; z=c, w=0.0f; };
//	inline __host__ __device__ float4p(float a,float b,float c,float d) { x=a; y=b; z=c, w=d; };
//};

__global__ void time_out(void)
{
	// commit suicide
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int sum =id;
	while (1) sum ++;
}

__global__ void setup_randstates(curandState *state,long long seed)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init(seed,id,0,&state[id]);
}

// basic test function here
__global__ void generate_kernel1(curandState *state,float *buf)
{
	//int id = threadIdx.x + blockIdx.x*blockDim.x;
	int id = threadIdx.x;

	int offset = blockIdx.x*blockDim.x*SSlot*MSlot;

	curandState localState = state[id];

	/* Generate pseudo-random floats */
	for (int i = 0; i < SSlot*MSlot; i++) {
		buf[offset+id] = curand_uniform(&localState);  //adjacent threads access adjacent memory locations
		offset += blockDim.x;
	}
	/* Copy state back to global memory */
	state[id] = localState;
}

inline __host__ __device__ float get_comp(float4 *a,int i)
{
	return *((float *)(a)+i);
}

inline __host__ __device__ void put_comp(float4 *a,int i,float val)
{
	*((float *)(a)+i) = val;
}

// histogram hit points
__device__ void add_ahit(uint *ahit,float4 &hit,float4 &limits,int nxy,int nz)
{
		int ix = (int)((float)nxy*((hit.x-limits.x)/(limits.y-limits.x)));  //same limits for x and y
		int iy = (int)((float)nxy*((hit.y-limits.x)/(limits.y-limits.x)));
		int iz = (int)((float)nz *((hit.z-limits.z)/(limits.w-limits.z)));
		ix = min(ix,nxy-1); ix = max(ix,0);
		iy = min(iy,nxy-1); iy = max(iy,0);
		iz = min(iz,nz-1);  iz = max(iz,0);
		atomicAdd(ahit+(iz*nxy+iy)*nxy+ix,1);
}

__global__ void generate_ahits(float4 *buf,uint *ahit,float4 limits,int nxy,int nz)
{
	//int id = threadIdx.x + blockIdx.x*blockDim.x;
	int ndo = (NThreads*NBlocks*MSlot)/(blockDim.x*gridDim.x);
	int istart = SSlot*(threadIdx.x + blockIdx.x*blockDim.x*ndo);
	for (int k=0; k<ndo; k++){		
		float4 a = buf[istart];
		float4 n = buf[istart+1];
		for (int j = 2; j<6; j+=2){
			if (buf[istart+j].x > 0.0f && buf[istart+j].z >0.0f){ //gamma1
				float4 hit = a + buf[istart+j].x*n;
				add_ahit(ahit,hit,limits,nxy,nz);
				hit	= a + buf[istart+j].z*n;
				add_ahit(ahit,hit,limits,nxy,nz);
			}
			if (buf[istart+j+1].x > 0.0f && buf[istart+j+1].z >0.0f){  //gamma2
				float4 hit = a - buf[istart+j+1].x*n;
				add_ahit(ahit,hit,limits,nxy,nz);
				hit	= a - buf[istart+j+1].z*n;
				add_ahit(ahit,hit,limits,nxy,nz);
			}
		}
		istart +=SSlot*blockDim.x;
	}
}

// histogram decay points
__global__ void generate_avol(float4 *buf,uint *avol,float4 limits,int nxy,int nz)
{
	//int id = threadIdx.x + blockIdx.x*blockDim.x;
	int ndo = (NThreads*NBlocks*MSlot)/(blockDim.x*gridDim.x);
	int istart = SSlot*(threadIdx.x + blockIdx.x*blockDim.x*ndo);
	for (int k=0; k<ndo; k++){
		int ix = (int)((float)nxy*((buf[istart].x-limits.x)/(limits.y-limits.x)));  //same limits for x and y
		int iy = (int)((float)nxy*((buf[istart].y-limits.x)/(limits.y-limits.x)));
		int iz = (int)( (float)nz*(buf[istart].z-limits.z)/(limits.w-limits.z) );

		ix = min(ix,nxy-1); ix = max(ix,0);
		iy = min(iy,nxy-1); iy = max(iy,0);
		iz = min(iz,nz-1); iz = max(iz,0);
		atomicAdd(avol+(iz*nxy+iy)*nxy+ix,1);
		//avol[(iz*nxy+iy)*nxy+ix] += 1;
		//if (id==511) printf("id %d istart %d ndo %d bdx %d gdx %d x %8.3f y %8.3f z %8.3f => %d %d %d\n",id,istart,ndo,blockDim.x,gridDim.x,buf[istart].x,buf[istart].y,buf[istart].z,ix,iy,iz);
		istart +=SSlot*blockDim.x;
	}
}


// histogram decay points with special treatment at max & min z bins using 48 ring small detector
__global__ void generate_avol_phantom(float4 *buf,uint *avol,float4 limits,int nxy,int nz)
{
	//int id = threadIdx.x + blockIdx.x*blockDim.x;
	int ndo = (NThreads*NBlocks*MSlot)/(blockDim.x*gridDim.x);
	int istart = SSlot*(threadIdx.x + blockIdx.x*blockDim.x*ndo);
	for (int k=0; k<ndo; k++){
		int ix = (int)((float)nxy*((buf[istart].x-limits.x)/(limits.y-limits.x)));  //same limits for x and y
		int iy = (int)((float)nxy*((buf[istart].y-limits.x)/(limits.y-limits.x)));
		if (buf[istart].z >= limits.z-0.5f*F120_ZBin && buf[istart].z <= limits.w+0.5f*F120_ZBin) { // cut if outside small detector
			int iz = (int)((float)nz*(buf[istart].z-limits.z)/(limits.w-limits.z));
			ix = min(ix,nxy-1); ix = max(ix,0);
			iy = min(iy,nxy-1); iy = max(iy,0);
			iz = min(iz,nz-1); iz = max(iz,0);
			atomicAdd(avol+(iz*nxy+iy)*nxy+ix,1);
		}
		istart +=SSlot*blockDim.x;
	}
}

__device__ float phi_at_hit(float4 &a,float4 &n,float lam)
{
	float4 hit = a+lam*n;
	float phi = atan2f(hit.x,hit.y);
	if (phi<0.0f)phi += PI2;
	//int nphi = ((int)((F120_BPnum*phi)/PI2));
	//float phi2 = (float(nphi)+0.5f)*PI2 / (float)F120_BPnum;   // NB extra 1/2 block rotation

	return phi;
}

__device__ uint lam2crystal_dev(float4 &a, float4 &n, float lam, float *c)
{
	float4 hit = a+lam*n;

	float phi = atan2f(hit.x,hit.y);
	if (phi<0.0f)phi += PI2;
	
	int nphi = ((int)((F120_BPnum*phi)/PI2));       // this is block number - case phi=0 inter-block gap
	//int nphi = ((int)((F120_BPnum*phi)/PI2) + 0.5f);  // this is block number - case phi=0 centre of block
	//nphi = nphi%F120_BPnum;                           // necessary for        - case phi=0 centre of block

	float phi2 = (float(nphi)+0.5f)*PI2 / (float)F120_BPnum;    //NB phi=0 at gap between blocks  (0.5f 1/2 block rotn)
	//float phi2 = (float(nphi)    )*PI2 / (float)F120_BPnum;       //  phi=0 at centre of block (14/11/17)

	float4 hit2;
	hit2.x = hit.x*cosf(phi2) - hit.y*sinf(phi2);
	hit2.y = hit.x*sinf(phi2) + hit.y*cosf(phi2);
	hit2.z = hit.z;
	hit2.w = 0.0f;

	// ix is aziimutal position around ring (21 bits  need ~9)
	int ix = (int)((float)F120_Cnum*(hit2.x - c[0]) / (c[1] - c[0]));
	ix = max(0,ix); ix = min(F120_Cnum-1,ix);
	ix += F120_Cnum*nphi;

	// iy is undetected interaction depth in crystal (4 bits) - might be worth keeping in MC
	int iy = (int)((float)F120_NDoi*(hit2.y - c[2]) / (c[3] - c[2]));
	iy = max(0,iy); iy = min(F120_NDoi-1,iy);

	// iz is ring number (7 bits)
	int iz = (int)( (float)F120_NZ*(hit2.z - c[4])/(c[5] - c[4]) );
	iz = max(0,iz); iz = min(F120_NZ-1,iz);

	//  allow 7 bits for iz and 4 for iy        
	uint value = (ix << 11) + (iz << 4) + iy;

	return value;
}

//  folded amap - this version using extra doi buffer for end points 
__global__ void generate_fmap_doi(float4 *buf,float2 *dbuf,uint *amap,float *c,int mfold)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int steps = (NBlocks*NThreads*MSlot)/(blockDim.x*gridDim.x);

	int doffset = id*DSlot;
	int offset = id*SSlot;
	//int slice = F120_STride;
	int MFold_XY = (F120_NZ-mfold)*F120_NXY;
	//int MFold_Z = (mfold+1)*F120_NXY;
	//int MFold_STride = MFold_XY*MFold_Z;

	for (int k=0; k<steps; k++){
		float lam1 = dbuf[doffset].x;  // in face gamma1
		float lam2 = dbuf[doffset].y;  // in face gamma2
		if (lam1 > 0.0f && lam2 > 0.0f){
			float4 a = buf[offset];
			float4 n = buf[offset+1];
			float4 m = -n;  // reverse for 2nd gamma
			int p = lam2crystal_dev(a,n,lam1,c);
			int c1 = p >> 11;
			int z1 = (p & 0x07ff)>>4;   // keep low 11 bits then discard doi

			p = lam2crystal_dev(a,m,lam2,c);
			int c2 = p >> 11;
			int z2 = (p & 0x07ff)>>4;
			//    for slice (x1,z1) increment pixel (x2,z2)

			if (z1 > z2){    //  z1 < z2 if cross plane (TODO modify generator to enforce this)
				int tz = z1;
				int tc = c1;
				z1 = z2;
				c1 = c2;
				z2 = tz;
				c2 = tc;
			}
			if (z1 == z2 && c1 > c2){   // c1 < c2 if in plane
				int tc = c1;
				c1 = c2;
				c2 = tc;
			}
			z2 = max(0,z2-mfold);  // NB source zbin 47 or 48 only;
			z1 = min(z1,mfold+1);
			int index = (z1*F120_NXY+c1)*MFold_XY+(z2*F120_NXY)+c2;
			atomicAdd(&amap[index],1);

			// also  for slice (x2,z2) increment pixel (x1,z1)  Thus each lor apperars twice - useful for inspection
			//index = (z2*F120_NXY+x2)*F120_STride+(z1*F120_NXY)+x1;
			//inc = index & 0x0001 ? 65536 : 1; // old ushort accumulate - overflows
			//atomicAdd(&amap[index/2],inc);
		}
		offset+=SSlot*(blockDim.x*gridDim.x);
		doffset+=DSlot*(blockDim.x*gridDim.x);
	}
}

// this version for vanilla true entry points but without doi
// change notation 12/07/17 use c not x for crysal numbers and p for packed value!!!
__global__ void generate_fmap(float4 *buf,uint *amap,float *c,int mfold)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int steps = (NBlocks*NThreads*MSlot)/(blockDim.x*gridDim.x);

	int offset = id*SSlot;
	//int slice = F120_STride;
	int MFold_XY = (F120_NZ-mfold)*F120_NXY;
	//int MFold_Z = (mfold+1)*F120_NXY;
	//int MFold_STride = MFold_XY*MFold_Z;

	for (int k=0; k<steps; k++){
		float lam1 = buf[offset+2].x;  // in face gamma1
		float lam2 = buf[offset+3].x;  // in face gamma2
		if (lam1 > 0.0f && lam2 > 0.0f){
			float4 a = buf[offset];
			float4 n = buf[offset+1];
			float4 m = -n;  // reverse for 2nd gamma

			int p = lam2crystal_dev(a,n,lam1,c);
			int c1 = p >> 11;
			int z1 = (p & 0x07ff)>>4;

			p = lam2crystal_dev(a,m,lam2,c);
			int c2 = p >> 11;
			int z2 = (p & 0x07ff)>>4;

			if (z1 > z2){    //  z1 < z2 if cross plane (TODO modify generator to enforce this)
				int tz = z1;
				int tc = c1;
				z1 = z2;
				c1 = c2;
				z2 = tz;
				c2 = tc;
			}
			if (z1 == z2 && c1 > c2){   // c1 < c2 if in plane
				int tc = c1;
				c1 = c2;
				c2 = tc;
			}
			//if (z1 > 47)printf("z1 z2 (%d %d)\n",z1,z2);
			z2 = max(0,z2-mfold);  // NB zbin 47 or 48 only;
			z1 = min(z1,mfold+1);
			int index = (z1*F120_NXY+c1)*MFold_XY+(z2*F120_NXY)+c2;
			atomicAdd(&amap[index],1);
		}
		offset+=SSlot*(blockDim.x*gridDim.x);
	}
}

// this version using extra doi buffer for end points 
__global__ void generate_amap_doi(float4 *buf,float2 *dbuf,uint *amap,float *c)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int steps = (NBlocks*NThreads*MSlot)/(blockDim.x*gridDim.x);

	int doffset = id*DSlot;
	int offset = id*SSlot;
	//int slice = F120_STride;

	for (int k=0; k<steps; k++){
		float lam1 = dbuf[doffset].x;  // in face gamma1
		float lam2 = dbuf[doffset].y;  // in face gamma2
		if (lam1 > 0.0f && lam2 > 0.0f){
			float4 a = buf[offset];
			float4 n = buf[offset+1];
			float4 m = -n;  // reverse for 2nd gamma
			int p = lam2crystal_dev(a,n,lam1,c);
			int c1 = p >> 11;
			int z1 = (p & 0x7ff)>>4;   // keep low 11 bits then discard doi

			p = lam2crystal_dev(a,m,lam2,c);
			int c2 = p >> 11;			
			int z2 = (p & 0x7ff)>>4;

			//    for slice (x1,z1) increment pixel (x2,z2)
			int index = (z1*F120_NXY+c1)*F120_STride+(z2*F120_NXY)+c2;
			atomicAdd(&amap[index],1);

			// also  for slice (x2,z2) increment pixel (x1,z1)  Thus each lor apperars twice - useful for inspection
			//index = (z2*F120_NXY+x2)*F120_STride+(z1*F120_NXY)+x1;
			//inc = index & 0x0001 ? 65536 : 1; // old ushort accumulate - overflows
			//atomicAdd(&amap[index/2],inc);
		}
		offset+=SSlot*(blockDim.x*gridDim.x);
		doffset+=DSlot*(blockDim.x*gridDim.x);
	}
}

// this version for vanilla true entry points but without doi
__global__ void generate_amap(float4 *buf,uint *amap,float *c)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int steps = (NBlocks*NThreads*MSlot)/(blockDim.x*gridDim.x);

	int offset = id*SSlot;
	//int slice = F120_STride;

	for (int k=0; k<steps; k++){
		float lam1 = buf[offset+2].x;  // in face gamma1
		float lam2 = buf[offset+3].x;  // in face gamma2
		if (lam1 > 0.0f && lam2 > 0.0f){
			float4 a = buf[offset];
			float4 n = buf[offset+1];
			float4 m = -n;  // reverse for 2nd gamma

			int p = lam2crystal_dev(a,n,lam1,c);
			int c1 = p >> 11;
			int z1 = (p & 0x07ff)>>4;   // keep low 11 bits then discard doi bits
	
			p = lam2crystal_dev(a,m,lam2,c);
			int c2 = p >> 11;
			int z2 = (p & 0x07ff)>>4;

			//    for slice (x1,z1) increment pixel (x2,z2)
			int index = (z1*F120_NXY+c1)*F120_STride+(z2*F120_NXY)+c2;
			atomicAdd(&amap[index],1);

			// also  for slice (x2,z2) increment pixel (x1,z1)  Thus each lor apperars twice - useful for inspection
			//index = (z2*F120_NXY+x2)*F120_STride+(z1*F120_NXY)+x1;
			//inc = index & 0x0001 ? 65536 : 1;
			//atomicAdd(&amap[index/2],inc);
		}
		offset+=SSlot*(blockDim.x*gridDim.x);
	}
}


// this version for vanilla true entry points but without doi using ZDZ map for phantoms
__global__ void generate_zdzmap(float4 *buf,uint *amap,float *c)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int steps = (NBlocks*NThreads*MSlot)/(blockDim.x*gridDim.x);

	int offset = id*SSlot;
	//int slice = F120_STride;

	for (int k=0; k<steps; k++){
		float lam1 = buf[offset+2].x;  // in face gamma1
		float lam2 = buf[offset+3].x;  // in face gamma2
		if (lam1 > 0.0f && lam2 > 0.0f){
			float4 a = buf[offset];
			float4 n = buf[offset+1];
			float4 m = -n;  // reverse for 2nd gamma

			int p = lam2crystal_dev(a,n,lam1,c);
			int c1 = p >> 11;
			int z1 = (p & 0x07ff)>>4;   // keep low 11 bits then discard doi bits
	
			p = lam2crystal_dev(a,m,lam2,c);
			int c2 = p >> 11;
			int z2 = (p & 0x07ff)>>4;

			if (z1 > z2){
				int zt = z1;
				int ct = c1;
				z1 = z2;
				c1 = c2;
				z2 = zt;
				c2 = ct;
			}
			if (z1 == z2 && c1 > c2){
				int ct =c1;
				c1 = c2;
				c2 =ct;
			}
			z1 -= 24;  // bug fix 12/1017 was 23
			z2 -= 24;
			if (z1 >= 0 && z2 <48){
				int dz = z2-z1;
				int dc = abs(c2-c1);
				if (c1 > c2) dc = F120_NXY-dc;  // fix logically negative dc values
				dc-= F120_DCmin;
				if (dc >= 0 && dc <F120_DCsize){
					int dzoffset = z1 + (dz*(97-dz))/2;
					int index = (dc*F120_NXY+c1)*F120_DZstride+dzoffset;
					atomicAdd(&amap[index],1);
				}
			}
		}
		offset+=SSlot*(blockDim.x*gridDim.x);
	}
}

__device__ void hit_planes(float4 *b,float *c,float4 &a,float4 &n,float phi2,int tryflag)
{
//  This kernel applies phi2 rotn so that entry block plane is in x-z plane, i.e perp to y axis.
//	__shared__ float plam[192];  // NB 192 threads per block hardcoded (must be multiple of 6)
	__shared__ float plam[256];  // NB 256 threads per block hardcoded (must be multiple of 8)

	int t_id = threadIdx.x;
	//int c_id = t_id%6;
	int c_id = t_id%8;
	int r_id = c_id/2;
	plam[t_id] = -1.0f;

	if (tryflag && c_id<6){
		float a2[3];
		a2[0] = a.x*cosf(phi2) - a.y*sinf(phi2);
		a2[1] = a.x*sinf(phi2) + a.y*cosf(phi2);
		a2[2] = a.z;
		float n2[3];
		n2[0] = n.x*cosf(phi2) - n.y*sinf(phi2);
		n2[1] = n.x*sinf(phi2) + n.y*cosf(phi2);
		n2[2] = n.z;	

		// check 6 planes with 6 threads here
		float lam3 = (c[c_id] - a2[r_id]) / n2[r_id]; // this only works implicity if corners and normals in correct sequence	

		if (lam3 > 0.0f){
			float hx = a2[0]+lam3*n2[0];
			float hy = a2[1]+lam3*n2[1];
			float hz = a2[2]+lam3*n2[2];
			if (hx >= c[6] && hx <= c[7] && hy >= c[8] && hy <= c[9] && hz >= c[10] && hz <= c[11]) plam[t_id] = lam3;
		}
	}

	//__syncthreads();  // this sync caused a lot of pain, implies call to this function CANNOT be conditional, hence use tryflag.
	// in the end gave up and use 8 theads instead of 6. 32 threads per warp =8*4 => implict syncthreads. No time penalty on 970 card.

	// 6fold reduce using one thread in 8 

	if (c_id == 0){
		if (tryflag > 0){
			int good = 0;
			float lam4 = -1.0f;
			int ilam4 = -1;
			float lam5 = -1.0f;
			int ilam5 = -1;
			for (int k = 0; k<6; k++) if (plam[t_id+k]>0.0f){
				good++;
				if (lam4 < 0.0f){
					lam4 = plam[t_id+k];
					ilam4 = k;
				}
				else{
					lam5 = plam[t_id+k];
					ilam5 = k;
				}
			}
			// fix 0<lam4<lam5
			if (lam5 > 0.0f && lam4 > lam5){
				float t = lam4;
				lam4 = lam5;
				lam5 = t;
				int it = ilam4;
				ilam4 = ilam5;
				ilam5 = it;
			}
			if(good > 1)b[0] = make_float4(lam4,(float)ilam4,lam5,(float)ilam5);  // test good added 14/07/17
			else        b[0] = make_float4(-1.0f,-1.0f,-1.0f,-1.0f);
		}  // end tryflag = 1 
	}

	return;
}

__device__ float nudge(float4 &a,float4 &n,float lam)
{
	float hx = a.x+n.x*lam;
	float hy = a.y+n.y*lam;
	float cross_z = hx*n.y-hy*n.x;

	return  cross_z < 0.0f ? PI2/(float)F120_BPnum : -PI2/(float)F120_BPnum;
}

__device__ void set_hits(float4 *buf,float *c,float4 &a,float4 &n,float lam,int h2)
{
	// add flag to hit_planes call and remove if
	float phi2 = 0.0f;
	if (lam <= 0.0f) {
		//buf[0] = make_float4(-1.0f,-1.0f,-1.0f,-1.0f);  // just in case
		hit_planes(buf,c,a,n,0.0f,0);   // avoid race
	}
	else{  //block1
		float phi =  phi_at_hit(a,n,lam);
	
		int nphi = ((int)((F120_BPnum*phi)/PI2));              // block number
		phi2 = ((float)(nphi)+0.5f)*PI2 / (float)F120_BPnum;   //NB 0.5f 1/2 block rotn phi=0 between blocks


		//int nphi = ((int)( (F120_BPnum*phi)/PI2) +0.5f);      // 08/11/17 shift gap to 7.5 degrees
		//nphi = nphi%F120_BPnum;
		//phi2 = float(nphi)*PI2 / (float)F120_BPnum;           //  phi=0 at centre of block (14/11/17) 
		
		hit_planes(buf,c,a,n,phi2,1);
	}

	if (lam <= 0.0f || buf->w >=3.0f) {  // dummy call if exit block 1 ttop or z-limits
		buf += h2;
		//buf[0] = make_float4(-1.0f,-1.0f,-1.0f,-1.0f);  // just in case
		hit_planes(buf,c,a,n,0.0f,0);   // avoid race
	}
	else{          // try again if ray misses or exits block1 through sides (i.e. buf.w = 0,1 or 2)
		//if (buf->w >= 2.0f)tryflag = 0;
		phi2 += nudge(a,n,buf->z);
		if (buf->w >= 0.0f && buf->w < 2.0f )buf += h2;  // second block hit if exit sides of block1
		hit_planes(buf,c,a,n,phi2,1);
	}
}

//   Block numbering inter-block gap at phi=0 if 0.5f addition in line 528
//
//             \          23 |  0                    
//              \      22    |     1       /               
//                \  21      |       2   /                 
//                  \        |         /                  
//               20   \      |       /    3               
//            19       \     |     /         4           
//                      \    |   /                     
//           18            \ | /             5
//---------------------------+----------------------------------                     
//                         / | \              6       
//           17          /   |   \                   
//                     /     |     \        7         
//           16      /       |        \   8           
//                 /         |          \            
//           15  /           |          9  \          
//             /   14        |               \       
//           /       13      |         10        \     
//          /          12    |    11



// track rays though prime & second block - clean and lean version using fuction call
__global__ void generate_kernel3(float4 *buf,float *c)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	//int offset = SSlot*(id/6);  // NB this offset shared by 6 threads (production)
	int offset = SSlot*(id/8);  // NB this offset shared by 8 threads force all in same warp so syncthreads implict
	//int t_id = threadIdx.x;

	float4 a = buf[offset];		
	float4 n = buf[offset+1];

	float lam = a.w;
	a.w = 0.0f;
	n.w = 0.0f;
	//if (lam > 0.0f)set_hits(buf+offset+2,c,a,n,lam,2,t_id);  // possible race condition at synthreads
	set_hits(buf+offset+2,c,a,n,lam,2);

	//__syncthreads();

	n = -buf[offset+1];
	lam = n.w;
	n.w = 0.0f;	
	//if (lam > 0.0f) set_hits(buf+offset+3,c,a,n,lam,2,t_id);
	set_hits(buf+offset+3,c,a,n,lam,2);
}

// use decay points for lor measured maps
__global__ void generate_kernel5(curandState *state,float4 *buf,float2 *dbuf)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	curandState localState = state[id];

	int steps = (NBlocks*NThreads*MSlot)/(blockDim.x*gridDim.x);

	int boffset = id*SSlot;
	int doffset = id;
	float scale = 1.0f/F120_LSOattn;
	for (int k=0; k<steps; k++){

		dbuf[doffset] = make_float2(0.0f,0.0f);

		float dist1 = (buf[boffset+2].z-buf[boffset+2].x);
		float dist2 = (buf[boffset+4].z-buf[boffset+4].x);
		float dist = dist1+dist2;
		if (dist > 0.0f){
			// generate exponential path here		
			float path = -scale*logf(1.0f-curand_uniform(&localState)); 
			if (path < dist1)      dbuf[doffset].x = buf[boffset+2].x + path;
			else if (path < dist) dbuf[doffset].x = buf[boffset+4].x + path - dist1;
		}

		dist1 = (buf[boffset+3].z-buf[boffset+3].x);
		dist2 = (buf[boffset+5].z-buf[boffset+5].x);
		dist = dist1+dist2;
		if (dist > 0.0f){
			// generate exponential path here
			float path = -scale*logf(1.0f-curand_uniform(&localState));
			if (path < dist1)      dbuf[doffset].y = buf[boffset+3].x + path;
			else if (path < dist) dbuf[doffset].y = buf[boffset+5].x + path - dist1;
		}
		boffset += SSlot*(blockDim.x*gridDim.x);
		doffset +=       (blockDim.x*gridDim.x);
	}
	state[id] = localState;
}

// use entry points for ideal geometrical lor maps
__global__ void generate_kernel4(float4 *buf,int4 *cry,float *c)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int steps = (NBlocks*NThreads*MSlot)/(blockDim.x*gridDim.x);

	int boffset = id*SSlot;
	int coffset = id*CSlot;

	for (int k=0;k<steps;k++){
		float4 a = buf[boffset];
		float4 n = buf[boffset+1];
		float4 m = -n;  // reverse for 2nd gamma
		cry[coffset] = make_int4(-1,-1,-1,-1);
		// first hit block
		if (buf[boffset+2].x > 0.0f) cry[coffset].x = lam2crystal_dev(a,n, buf[boffset+2].x,c);
		if (buf[boffset+2].z > 0.0f) cry[coffset].y = lam2crystal_dev(a,n, buf[boffset+2].z,c);
		if (buf[boffset+3].x > 0.0f) cry[coffset].z = lam2crystal_dev(a,m, buf[boffset+3].x,c);
		if (buf[boffset+3].z > 0.0f) cry[coffset].w = lam2crystal_dev(a,m, buf[boffset+3].z,c);
		// second hit block
		cry[coffset+1] = make_int4(-1,-1,-1,-1);
		if (buf[boffset+4].x > 0.0f) cry[coffset+1].x = lam2crystal_dev(a,n, buf[boffset+4].x,c);
		if (buf[boffset+4].z > 0.0f) cry[coffset+1].y = lam2crystal_dev(a,n, buf[boffset+4].z,c);
		if (buf[boffset+5].x > 0.0f) cry[coffset+1].z = lam2crystal_dev(a,m, buf[boffset+5].x,c);
		if (buf[boffset+5].z > 0.0f) cry[coffset+1].w = lam2crystal_dev(a,m, buf[boffset+5].z,c);
		boffset+=SSlot*(blockDim.x*gridDim.x);
		coffset+=CSlot*(blockDim.x*gridDim.x);
	}
}

// flag buf empty
__global__ void fill_bigbuf(float4 *buf,float val)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	//int offset = id*SSlot*MSlot;
	while (id < BSize){
		buf[id] = make_float4(val,val,val,val);
		id += stride;
	}
	//for (int k=0; k<SSlot*MSlot; k++) buf[offset+k] = make_float4(val,val,val,val);
}

// generate rays and find prime blocks allow  source to be cylinder sphere or cuboid
// TODO add complex sources??
__global__ void generate_kernel2(curandState *state,float4 *buf,Source s)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int offset = id*SSlot*MSlot;
	
	curandState localState = state[id];

	int done = 0;
	int trys = 0;
	float4 a;
	float4 n;

	float vcos = cosf(s.o.w);  // added 08/11/17 to rotate grid
	float vsin = sinf(s.o.w);

	//if (id==0) printf("rotation %f %f %f\n",s.o.w,vcos,vsin);
	//if (id==0) printf("Rmin %.3f XYBin %.3f SNZ2 %d\n",F120_Rmin,F120_XYBin,F120_SNZ2);

	int max_trys = 20;  // timeout after 20 tries to get hit added 30/06/17 + need to count good and bad
	int tot_trys=0;
	int ngood = 0;
	int nbad = 0;
	while(done < MSlot){
		trys++;
		tot_trys++;
		if (s.t == CYLINDER){
			float phi2 = curand_uniform(&localState)*PI2;
			float r = sqrtf(curand_uniform(&localState))*s.dim.x;  // square root here	
			a.x = r*sinf(phi2) + s.o.x;
			a.y = r*cosf(phi2) + s.o.y;
			a.z = (curand_uniform(&localState)-0.5f)*s.dim.y + s.o.z;	
		}
		else if (s.t == SPHERE){
			float phi2 = curand_uniform(&localState)*PI2;
			float ctheta = 2.0f*curand_uniform(&localState) - 1.0f;
			float stheta = sinf(acosf(ctheta));
			float r = cbrtf(curand_uniform(&localState))*s.dim.x;  // cube root here	
			a.x = r*sinf(phi2)*stheta + s.o.x;
			a.y = r*cosf(phi2)*stheta + s.o.y;
			a.z = r*ctheta + s.o.z;	
		}
		else if (s.t == VOXEL){
			a.x = (curand_uniform(&localState)-0.5f)*s.dim.x + s.o.x;
			a.y = (curand_uniform(&localState)-0.5f)*s.dim.y + s.o.y;
			a.z = (curand_uniform(&localState)-0.5f)*s.dim.z + s.o.z;
		}

		else if (s.t == HALFVOX){
			a.x = (curand_uniform(&localState)-0.5f)*s.dim.x + s.o.x;
			a.y = (curand_uniform(&localState)-0.5f)*s.dim.y + s.o.y;
			a.z = (curand_uniform(&localState)-0.5f)*s.dim.z + s.o.z;
			if (a.x < a.y){
				float t = a.x;
				a.x = a.y;
				a.y = t;
			}
		}

		if (s.o.w != 0.0f){  //modifed 17/11/17  always rotate
			float xt = a.x;
			float yt = a.y;
			a.x = xt*vcos - yt*vsin;  //rotate object anticlockwise
			a.y = xt*vsin + yt*vcos;
		}


		// phi is uniform in x-y plane of scanner
		float phi = curand_uniform(&localState)*PI2;
		// cos(theta) is uniform and is angle wrt to z axis of scanner
		//float theta = acosf(0.05f*(1.0f-2.0f*curand_uniform(&localState)));
		//float theta = acosf(F120_Thetacut*(1.0f-2.0f*curand_uniform(&localState)));
		float theta = acosf(1.0f-2.0f*curand_uniform(&localState));
		n.x = sinf(phi)*sinf(theta);
		n.y = cosf(phi)*sinf(theta);
		n.z = cosf(theta);
		//if (fabs(theta) < 0.005f) printf("tiny theta %f z %f\n",theta,a.z);  // result OK


		//n.x = -sinf(phi)*sinf(theta);  // debug switching gammas
		//n.y = -cosf(phi)*sinf(theta);
		//n.z = -cosf(theta);

		float A = n.x*n.x + n.y*n.y;
		float B = a.x*n.x + a.y*n.y;
		float C = a.x*a.x + a.y*a.y - F120_Rmin*F120_Rmin;
		float D = B*B-A*C;
		float rad = sqrtf(D);
		float lam1 = (-B+rad)/A;
		float z1 = a.z+lam1*n.z;
		float lam2 = (-B-rad)/A;
		float z2 = a.z+lam2*n.z;
		//if (z1 <= F120_BZface && z1 >= 0.0f && z2 <= F120_BZface && z2 >= 0.0f && fabs(z1-z2) <= F120_TrueZface){  // check z ok, but might still go through gap...
		if (z1 <= F120_BZface && z1 >= 0.0f && z2 <= F120_BZface && z2 >= 0.0f ){  // check z ok, but might still go through gap...
			done++;
			a.w = lam1;
			buf[offset] =a;
			n.w = lam2;
			buf[offset+1] =n;
			offset+=SSlot;
			trys = 0;
			ngood++;
		}
		else nbad++;
		if (trys > max_trys){  // deal with timeout here 
			buf[offset] =   make_float4(-1.0f,-1.0f,-1.0f,-1.0f);
			buf[offset+1] = make_float4( 1.0f, 1.0f, 1.0f, 1.0f);  // will be reversed in kernel3
			offset+=SSlot;  // added 14/07/17
			done++;
			trys = 0;  // allows 20 trys for EACH entry
		}
	}

	/* Copy state back to global memory */
	devglob_gentrys[id].x += ngood;
	devglob_gentrys[id].y += nbad;
	devglob_gentrys[id].z += tot_trys;

	state[id] = localState;
}


__global__ void generate_kernel2_phantom(curandState *state,float4 *buf,Source s,uint *avol,float4 limits,int nxy,int nz)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int offset = id*SSlot*MSlot;
	
	curandState localState = state[id];

	int done = 0;
	int trys = 0;
	float4 a;
	float4 n;

	float vcos = cosf(s.o.w);  // added 08/11/17 to rotate grid
	float vsin = sinf(s.o.w);

	//if (id==0) printf("rotation %f %f %f\n",s.o.w,vcos,vsin);

	int max_trys = 20;  // timeout after 20 tries to get hit added 30/06/17 + need to count good and bad
	int tot_trys=0;
	int ngood = 0;
	int nbad = 0;
	while(done < MSlot){
		trys++;
		tot_trys++;
		if (s.t == CYLINDER){
			float phi2 = curand_uniform(&localState)*PI2;
			float r = sqrtf(curand_uniform(&localState))*s.dim.x;  // square root here	
			a.x = r*sinf(phi2) + s.o.x;
			a.y = r*cosf(phi2) + s.o.y;
			a.z = (curand_uniform(&localState)-0.5f)*s.dim.y + s.o.z;	
		}
		else if (s.t == SPHERE){
			float phi2 = curand_uniform(&localState)*PI2;
			float ctheta = 2.0f*curand_uniform(&localState) - 1.0f;
			float stheta = sinf(acosf(ctheta));
			float r = cbrtf(curand_uniform(&localState))*s.dim.x;  // cube root here	
			a.x = r*sinf(phi2)*stheta + s.o.x;
			a.y = r*cosf(phi2)*stheta + s.o.y;
			a.z = r*ctheta + s.o.z;	
		}
		else if (s.t == VOXEL){
			a.x = (curand_uniform(&localState)-0.5f)*s.dim.x + s.o.x;
			a.y = (curand_uniform(&localState)-0.5f)*s.dim.y + s.o.y;
			a.z = (curand_uniform(&localState)-0.5f)*s.dim.z + s.o.z;
		}
	
		else if (s.t == HALFVOX){
			a.x = (curand_uniform(&localState)-0.5f)*s.dim.x + s.o.x;
			a.y = (curand_uniform(&localState)-0.5f)*s.dim.y + s.o.y;
			a.z = (curand_uniform(&localState)-0.5f)*s.dim.z + s.o.z;
			if (a.x < a.y){
				float t = a.x;
				a.x = a.y;
				a.y = t;
			}
		}

		if (s.o.w != 0.0f){  //modifed 17/11/17  always rotate
			float xt = a.x;
			float yt = a.y;
			a.x = xt*vcos - yt*vsin;  //rotate object anticlockwise
			a.y = xt*vsin + yt*vcos;
		}

		//  for phantom save all generated x-y-z
		int ix = (int)((float)nxy*((a.x-limits.x)/(limits.y-limits.x)));  //same limits for x and y
		int iy = (int)((float)nxy*((a.y-limits.x)/(limits.y-limits.x)));
		int iz = (int)((float)nz*(a.z-limits.z)/(limits.w-limits.z) );
		ix = min(ix,nxy-1); ix = max(ix,0);
		iy = min(iy,nxy-1); iy = max(iy,0);
		iz = min(iz,nz-1); iz = max(iz,0);
		atomicAdd(avol+(iz*nxy+iy)*nxy+ix,1);

		// phi is uniform in x-y plane of scanner
		float phi = curand_uniform(&localState)*PI2;
		float theta = acosf(1.0f-2.0f*curand_uniform(&localState));
		n.x = sinf(phi)*sinf(theta);
		n.y = cosf(phi)*sinf(theta);
		n.z = cosf(theta);

		float A = n.x*n.x + n.y*n.y;
		float B = a.x*n.x + a.y*n.y;
		float C = a.x*a.x + a.y*a.y - F120_Rmin*F120_Rmin;
		float D = B*B-A*C;
		float rad = sqrtf(D);
		float lam1 = (-B+rad)/A;
		float z1 = a.z+lam1*n.z;
		float lam2 = (-B-rad)/A;
		float z2 = a.z+lam2*n.z;

		if (z1 <= F120_BZface && z1 >= 0.0f && z2 <= F120_BZface && z2 >= 0.0f ){  // check z ok, but might still go through gap...
			done++;
			a.w = lam1;
			buf[offset] =a;
			n.w = lam2;
			buf[offset+1] =n;
			offset+=SSlot;
			trys = 0;
			ngood++;
		}
		else nbad++;
		if (trys > max_trys){  // deal with timeout here 
			buf[offset] =   make_float4(-1.0f,-1.0f,-1.0f,-1.0f);
			buf[offset+1] = make_float4( 1.0f, 1.0f, 1.0f, 1.0f);  // will be revered in kernel3
			offset+=SSlot;  // added 14/07/17
			done++;
			trys = 0;  // allows 20 trys for EACH entry
		}
	}

	/* Copy state back to global memory */
	devglob_gentrys[id].x += ngood;
	devglob_gentrys[id].y += nbad;
	devglob_gentrys[id].z += tot_trys;

	state[id] = localState;
}



__global__ void generate_kernel0(int n)
{
	__shared__ int item[513];
	int tid = threadIdx.x;
	//int sum2 = 0;
	item[tid] = 0;
	//__syncthreads();
	if (tid>200) {
		for (int j = 0; j<50*tid; j++)item[tid] += 1;
		item[512] += item[tid];
	}

	item[tid] = blockIdx.x+1;
	__syncthreads();
	if (tid == 0){
		int sum = 0;
		for (int k = 0; k<512; k++) sum += item[k];
		printf("Block %d sum %d sum2 %d\n",blockIdx.x,sum,item[512]);
	}
}


#if 0
__global__ void clumsum(int *f,int *r)
{
	__shared__ int s[256];
	int i = threadIdx.x;
	int j= i+1;
	s[i] = f[i];   // copy to shared memory
	__syncthreads();

	// local sum reduce operation 
	if (!(j & 0x0001)) s[i] +=	s[i-1];
	__syncthreads();
	if (!(j & 0x0003)) s[i] +=	s[i-2];
	__syncthreads();
	if (!(j & 0x0007)) s[i] +=	s[i-4];
	__syncthreads();
	if (!(j & 0x000f)) s[i] +=	s[i-8];
	__syncthreads();
	if (!(j & 0x001f)) s[i] +=	s[i-16];
	__syncthreads();
	if (!(j & 0x003f)) s[i] +=	s[i-32];
	__syncthreads();
	if (!(j & 0x007f)) s[i] +=	s[i-64];
	__syncthreads();
	if (!(j & 0x00ff)) s[i] +=	s[i-128];
	__syncthreads();

	// cumlulative sum up to i'th
	int c = s[i];
	if ((j>2) && (j & 0x01)) c += s[(j & 0xfffe)-1];
	if ((j>4) && (j & 0x02)) c += s[(j & 0xfffc)-1]; 
	if ((j>8) && (j & 0x04)) c += s[(j & 0xfff8)-1];
	if ((j>16) && (j & 0x08)) c += s[(j & 0xfff0)-1];
	if ((j>32) && (j & 0x10)) c += s[(j & 0xffe0)-1];
	if ((j>64) && (j & 0x20)) c += s[(j & 0xffc0)-1];
	if ((j>128) && (j & 0x40)) c += s[(j & 0xff80)-1];

	r[i] = c;


}

#endif
 
#endif
