#ifndef LORS_H_
#define LORS_H_

//#include "scanner.h"

#ifndef uint
typedef  unsigned char uchar;
typedef  unsigned short ushort;
typedef  unsigned int  uint;
#endif

// on refelction this should be a proper class with methods.

struct Lor {
	uint key;
	float val;
};

struct NLtabheader
{
	size_t lors;
	float norm;	
};

struct quad {   // introduced 10/08/17 to simplify interface
	int z1;
	int c1;
	int z2;
	int c2;
};

struct hex {   // extended 21/08/17 to include voxel x,y for 8-fold sym operations
	int z1;
	int c1;
	int z2;
	int c2;
	int x;
	int y;
};

void lor_from(uint key,int &z1,int &c1,int &z2,int &c2);
void lor_from(uint key,quad &p);

bool lorsort(const Lor &lhs,const Lor &rhs) { return lhs.val > rhs.val; }    // > means decending  i.e lhs comes before rhs
bool lorsort_key(const Lor &lhs,const Lor &rhs) { return lhs.key > rhs.key; }    // > means decending
bool lorsort_z1z2(const Lor &lhs,const Lor &rhs) { 
	quad a;
	lor_from(lhs.key,a);
	uint keya = (a.z1*F120_NZ+a.z2)*F120_NXY*F120_NXY+(a.c1*F120_NXY+a.c2);
	quad b;
	lor_from(rhs.key,b);
	uint keyb = (b.z1*F120_NZ+b.z2)*F120_NXY*F120_NXY+(b.c1*F120_NXY+b.c2);
	return keya > keyb; 
}    // > means decending

//  flip p => q, return 0/1 if q proper/improper
int flip(quad &p, quad &q)
{
	q.z1 = p.z2;
	q.z2 = p.z1;
	q.c1 = p.c2;
	q.c2 = p.c1;
	if (q.z1 > q.z2) return 1;
	if (q.z1==q.z2 && q.c1 > q.c2) return 1;
	return 0;
}

int quads_same(quad &p, quad &q)
{
	if (p.z1 != q.z1 || p.z2 != q.z2 || p.c1 != q.c1 || p.c2 != q.c2) return 0;
	return 1;
}

uint lorval(uint *map,int z1,int c1,int z2,int c2)
{
	//z2 = max(0,z2);
	return map[(z1*F120_NXY+c1)*F120_STride+(z2*F120_NXY+c2)];
}

uint lorval(uint *map,quad &p)
{
	//z2 = max(0,z2);
	return map[(p.z1*F120_NXY+p.c1)*F120_STride+(p.z2*F120_NXY+p.c2)];
}

uint lorval(uint *map,int z1,int c1,int z2,int c2,int mfold)
{
	if (z2 < mfold || z1 > mfold+1) return 0;

	int MFold_STride = (F120_NZ-mfold)*F120_NXY;
	z2 -= mfold;
	return map[(z1*F120_NXY+c1)*MFold_STride+(z2*F120_NXY+c2)];
}

uint lorval(uint *map,quad &p,int mfold)
{
	if (p.z2 < mfold || p.z1 > mfold+1) return 0;

	int MFold_STride = (F120_NZ-mfold)*F120_NXY;
	p.z2 -= mfold;
	return map[(p.z1*F120_NXY+p.c1)*MFold_STride+(p.z2*F120_NXY+p.c2)];
}

// 
// pack lor into 32-bit uint key, note proper lors are sorted so that z1 <= z2 w.l.g
// | z1 | C1 | z2 | c2 |
// | 7  | 9  |  7 |  9 |

//get key from lor
uint key_from(int z1,int c1,int z2,int c2)
{
	//allow 7-bits for z and 9-bits for c, total 32 bits
	uint key = z1;
	key = (key<<9)+(uint)c1;
	key = (key<<7)+(uint)z2;
	key = (key<<9)+(uint)c2;
	return key;
}

uint key_from(quad &p)
{
	//allow 7-bits for z and 9-bits for c, total 32 bits
	uint key = p.z1;
	key = (key<<9)+(uint)p.c1;
	key = (key<<7)+(uint)p.z2;
	key = (key<<9)+(uint)p.c2;
	return key;
}

// make proper ordered lor, flag if change made
int proper_lor(int &z1,int &c1,int &z2,int &c2)
{
	if(z1 < z2) return 0;
	else if (z1 == z2 && c1 <= c2) return 0;
	int zt = z1;
	z1 = z2;
	z2 = zt;
	int ct = c1;
	c1 = c2;
	c2 = ct;
	return 1;
}

int proper_lor(quad &p)
{
	if(p.z1 < p.z2) return 0;
	else if (p.z1 == p.z2 && p.c1 <= p.c2) return 0;
	int zt = p.z1;  //or swap both z1/2 and c1/2 correct for both case z1=z2 and z2 <z1
	p.z1 = p.z2;
	p.z2 = zt;
	int ct = p.c1;
	p.c1 = p.c2;
	p.c2 = ct;
	return 1;
}

int is_improper_lor(int z1, int c1, int z2, int c2)
{
	if (z1 > z2) return 1;
	if (z2==z1 && c1 > c2) return 1;   // case z1==z2 && c1==c2 is an error
	return 0;  // require z1<z2 or if z1=z2 then c1 <c2
}

int is_improper_lor(quad &p)
{
	if (p.z1 > p.z2) return 1;
	if (p.z2==p.z1 && p.c1 > p.c2) return 1;   // case z1==z2 && c1==c2 is an error
	return 0;  // require z1<z2 or if z1=z2 then c1 <c2
}

int is_proper_lor(int z1, int c1, int z2, int c2)
{
	if (z1 < z2) return 1;
	if (z2==z1 && c1 < c2) return 1;   // case z1==z2 && c1==c2 is an error
	return 0;  // require z1<z2 or if z1=z2 then c1 <c2
}

int is_proper_lor(quad &p)
{
	if (p.z1 < p.z2) return 1;
	if (p.z2==p.z1 && p.c1 < p.c2) return 1;   // case z1==z2 && c1==c2 is an error
	return 0;  // require z1<z2 or if z1=z2 then c1 <c2
}

int is_improper_lor(uint key)
{
	quad p;
	lor_from(key,p);
	return is_improper_lor(p);
	//int z1,c1,z2,c2;
	//lor_from(key,z1,c1,z2,c2);
	//return is_improper_lor(z1,c1,z2,c2);
}

int is_proper_lor(uint key)
{
	quad p;
	lor_from(key,p);
	return is_proper_lor(p);
	//int z1,c1,z2,c2;
	//lor_from(key,z1,c1,z2,c2);
	//return is_proper_lor(z1,c1,z2,c2);
}

// get lor from key
void lor_from(uint key,int &z1,int &c1,int &z2,int &c2)
{
	//allow 7-bits for z and 9-bits for c, total 32 bits
	c2 = key & 0x000001ff;
	key = key>>9;
	z2 = key & 0x0000007f;
	key = key>>7;
	c1= key & 0x000001ff;
	key = key>>9;
	z1 = key;
	if (z1 >=F120_NZ) printf("KEY ERROR %d %d %d %d\n",z1,c1,z2,c2);
	return;
}

void lor_from(uint key,quad &p)
{
	//allow 7-bits for z and 9-bits for c, total 32 bits
	p.c2 = key & 0x000001ff;
	key = key>>9;
	p.z2 = key & 0x0000007f;
	key = key>>7;
	p.c1= key & 0x000001ff;
	key = key>>9;
	p.z1 = key;
	if (p.z1 >=F120_NZ) printf("KEY ERROR %d %d %d %d\n",p.z1,p.c1,p.z2,p.c2);
	return;
}

void lor_from(uint key,hex &p)
{
	//allow 7-bits for z and 9-bits for c, total 32 bits
	p.c2 = key & 0x000001ff;
	key = key>>9;
	p.z2 = key & 0x0000007f;
	key = key>>7;
	p.c1= key & 0x000001ff;
	key = key>>9;
	p.z1 = key;
	if (p.z1 >=F120_NZ) printf("KEY ERROR %d %d %d %d\n",p.z1,p.c1,p.z2,p.c2);
	return;
}

// these for small detector values using uncorrected sysmat
void small_lor_from(uint key,quad &p)
{
	lor_from(key,p);
	p.z1 -= F120_SM_Zshift;
	p.z2 -= F120_SM_Zshift;
	return;
}

void small_lor_from(uint key,hex &p)
{
	lor_from(key,p);
	p.z1 -= F120_SM_Zshift;
	p.z2 -= F120_SM_Zshift;
	return;
}

void small_lor_from(uint key,int &z1,int &c1,int &z2,int &c2)
{
	lor_from(key,z1,c1,z2,c2);
	z1 -= F120_SM_Zshift;
	z2 -= F120_SM_Zshift;
	return;
}

// index is positon in maps files
uint index_from_key(uint key)
{
	quad p;
	lor_from(key,p);
	uint index = (p.z1*F120_NXY+p.c1)*F120_STride+p.z2*F120_NXY+p.c2;
	return index;
}

uint index_from_lor(int z1,int c1,int z2,int c2)
{
	if (z1 <0 || z1 >= F120_NZ || z2 < 0 || z2 >= F120_NZ || c1 < 0 || c1 >= F120_NXY || c2 < 0 || c2 >= F120_NXY){
		printf("index_from_lor(%d,%d,%d,%d) error\n",z1,c1,z2,c2);
		return 0;
	}
	uint index = (z1*F120_NXY+c1)*F120_STride+z2*F120_NXY+c2;
	return index;
}

uint index_from_lor(quad &p)
{
	if (p.z1 <0 || p.z1 >= F120_NZ || p.z2 < 0 || p.z2 >= F120_NZ || p.c1 < 0 || p.c1 >= F120_NXY || p.c2 < 0 || p.c2 >= F120_NXY){
		//printf("index_from_lor(%d,%d,%d,%d) error\n",p.z1,p.c1,p.z2,p.c2);
		return 0;
	}
	uint index = (p.z1*F120_NXY+p.c1)*F120_STride+p.z2*F120_NXY+p.c2;
	return index;
}

uint index_from_lor(int z1,int c1,int z2,int c2,int mfold)
{
	if (mfold<=0) return index_from_lor(z1,c1,z2,c2);
	else if(z1 > (mfold+1) || z2 < mfold) { printf("error index_from_lor mfold %d called for z1/2 (%d %d\n",mfold,z1,z2); return 999999;}
	uint index = (z1*F120_NXY+c1)*(mfold+2)*F120_NXY+(z2-mfold)*F120_NXY+c2;
	return index;
}

uint index_from_lor(quad &p,int mfold)
{
	if (mfold<=0) return index_from_lor(p);
	else if(p.z1 > (mfold+1) || p.z2 < mfold) { printf("error index_from_lor mfold %d called for z1/2 (%d %d\n",mfold,p.z1,p.z2); return 999999;}
	uint index = (p.z1*F120_NXY+p.c1)*(mfold+2)*F120_NXY+(p.z2-mfold)*F120_NXY+p.c2;
	return index;
}


int lor_from_index(uint index,quad &p)
{
	p.c2 = index%F120_NXY; 	index /= F120_NXY;
	p.z2 = index%F120_NZ;	index /= F120_NZ;
	p.c1 = index%F120_NXY;
	p.z1 = index/F120_NXY;

	if (p.z1 <0 || p.z1 >= F120_NZ || p.z2 < 0 || p.z2 >= F120_NZ || p.c1 < 0 || p.c1 >= F120_NXY || p.c2 < 0 || p.c2 >= F120_NXY){
		printf("lor_from_index(%d,%d,%d,%d) error\n",p.z1,p.c1,p.z2,p.c2);
		return 1;
	}	
	return 0;
}

int lor_from_index(uint index, int &z1, int &c1, int &z2, int &c2)
{
	c2 = index%F120_NXY; 	index /= F120_NXY;
	z2 = index%F120_NZ;	index /= F120_NZ;
	c1 = index%F120_NXY;
	z1 = index/F120_NXY;

	if (z1 <0 || z1 >= F120_NZ || z2 < 0 || z2 >= F120_NZ || c1 < 0 || c1 >= F120_NXY || c2 < 0 || c2 >= F120_NXY){
		printf("lor_from_index(%d,%d,%d,%d) error\n",z1,c1,z2,c2);
		return 1;
	}	
	return 0;

}

// constuct mirror lor  (proper if input proper)
// NB mirror preserves delta_z
int mirror(quad &p, quad &m, int zbin)
{
	m.z2 = zbin-p.z1;
	m.z1 = zbin-p.z2;
	if (m.z1 == m.z2){   // no c's swop in degenerate case to preserve proper state
		m.c1 = p.c1;   
		m.c2 = p.c2;
	} 
	else {             // but swop c's in non degenerate cases
		m.c2 = p.c1;
		m.c1 = p.c2;
	}
	if      (p.z1 == p.z2 && p.z1 == m.z1 && p.z1 == m.z2)               return 1; // mirror identical to origin (z1=z2=48  & zbin = 96 only) Not an error
	else if (m.z2 >= F120_NZ || m.z2 < 0 || m.z1 >= F120_NZ || m.z1 < 0) return 2; // mirror outside detector - possible off axis and not an error in short detector
	return 0;
}

int mirror(hex &p, hex &m, int zbin)
{
	m.x = p.x;  // mirror and proper lors share same parent voxel. 
	m.y = p.y;
	m.z2 = zbin-p.z1;
	m.z1 = zbin-p.z2;
	if (m.z1 == m.z2){   // no c's swop in degenerate case to preserve proper state
		m.c1 = p.c1;   
		m.c2 = p.c2;
	} 
	else {             // but swop c's in non degenerate cases
		m.c2 = p.c1;
		m.c1 = p.c2;
	}
	if      (p.z1 == p.z2 && p.z1 == m.z1 && p.z1 == m.z2)               return 1; // mirror identical to origin (z1=z2=48  &xbin =96 only) Not an error
	else if (m.z2 >= F120_NZ || m.z2 < 0 || m.z1 >= F120_NZ || m.z1 < 0) return 2; // mirror outside detector - possible off axis
	return 0;
}

int mirror(hex &m,int zbin)
{
	hex p = m;
	//m.x = p.x;  // mirror and proper lors share same parent voxel. 
	//m.y = p.y;
	m.z2 = zbin-p.z1;
	m.z1 = zbin-p.z2;
	if (m.z1 == m.z2){   // no c's swop in degenerate case to preserve proper state
		m.c1 = p.c1;
		m.c2 = p.c2;
	}
	else {             // but swop c's in non degenerate cases
		m.c2 = p.c1;
		m.c1 = p.c2;
	}
	if (p.z1 == p.z2 && p.z1 == m.z1 && p.z1 == m.z2)               return 1; // mirror identical to origin (z1=z2=48  &xbin =96 only) Not an error
	else if (m.z2 >= F120_NZ || m.z2 < 0 || m.z1 >= F120_NZ || m.z1 < 0) return 2; // mirror outside detector - possible off axis
	return 0;
}

// NB decide key = 0 can flag error 10/08/17
uint key_from_mirror(quad &p,int zbin)
{
	quad m;
	if(mirror(p,m,zbin)) return 0;  // out of detector or identical to p

	return key_from(m);
}

uint key_from_mirror(int z1,int c1,int z2,int c2,int zbin)
{
	quad p = {z1,c1,z2,c2};
	return key_from_mirror(p,zbin);
}

int mirror_from_key(uint key, quad &m,int zbin)
{
	quad p;
	lor_from(key,p);
	return mirror(p,m,zbin);
}

int mirror_from_key(uint key, int &z1m, int &c1m, int &z2m, int &c2m,int zbin)
{
	quad p;
	lor_from(key,p);
	quad m;
	int check = mirror(p,m,zbin);
	z1m = m.z1;
	c1m = m.c1;
	z2m = m.z2;
	c2m = m.c2;
	return check;
}

#endif