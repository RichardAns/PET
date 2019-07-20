// based on CUDA float4 thisd  is a class with methods
// treats 4th component as padding and support [] addessing
#ifndef MYFLOAT4P_H
#define MYFLOAT4P_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <cuda.h> 
#include <helper_math.h>

class __device_builtin__ __builtin_align__(16) float4p{
public:
	float x,y,z,w;
	//copy constuctors used insted of make
	inline __host__ __device__ float4p() { x=y=z=w=0.0f; };
	inline __host__ __device__ float4p(const float4p &p) { x=p.x; y=p.y; z=p.z; w=p.w; };
	inline __host__ __device__ float4p(const float3 &p) { x=p.x; y=p.y; z=p.z; w=0.0f; };
	inline __host__ __device__ float4p(const float3 &p, const float q) { x=p.x; y=p.y; z=p.z; w=q; };

	inline __host__ __device__ float4p(const float &s) { x=y=z=w=s; };
	inline __host__ __device__ float4p(const float &a, const float &b, const float &c) { x=a; y=b; z=c; w=0.0f; };
	inline __host__ __device__ float4p(const float &a,const float &b,const float &c, const float &d) { x=a; y=b; z=c,w=d; };
	
	inline __host__ __device__ float4p(const int4 &p) { x=float(p.x); y=float(p.y); z=float(p.z); w=float(p.w); };
	inline __host__ __device__ float4p(const uint4 &p) { x=float(p.x); y=float(p.y); z=float(p.z); w=float(p.w); };
	
	//  member operator overloads
	inline __host__ __device__ float& operator[](int i) { return *((&x)+i); }  // added value here
	inline __host__ __device__ float4p operator+(const float4p &b) { return float4p(x + b.x, y + b.y, z + b.z, w + b.w); }
	inline __host__ __device__ float4p operator+(const float &b) { return float4p(x+b,y+b,z+b,w+b); }
	inline __host__ __device__ float4p operator+=(const float4p &b) { return float4p(x += b.x,y += b.y,z += b.z,w += b.w); }
	inline __host__ __device__ float4p operator+=(const float &b) { return float4p(x += b,y += b,z += b,w += b); }
	

	friend inline __host__ __device__ float4p operator+(const float4p &a,const float4p &b);
	friend inline __host__ __device__ float4p operator+(const float4p &a,const float &b);
	friend inline __host__ __device__ float4p operator+(const float &b,const float4p &a);

	friend inline __host__ __device__ float4p operator-(const float4p &a);
	friend inline __host__ __device__ float4p operator-(const float4p &a,const float4p &b);
	friend inline __host__ __device__ float4p operator-(const float4p &a,const float &b);
	friend inline __host__ __device__ float4p operator-(const float &b,const float4p &a);

	inline __host__ __device__ void operator-=(const float4p b)	{ x -= b.x; y -= b.y; z -= b.z; w -= b.w; }
	inline __host__ __device__ float4p operator-(const float b) { return float4p(x - b,y - b,z - b,w - b); }
	inline __host__ __device__ void operator-=(const float b) { x -= b; y -= b; z -= b; w -= b; }
	inline __host__ __device__ float4p operator*(const float4p b) {	return float4p(x*b.x, y*b.y, z*b.z, w*b.w); }
	inline __host__ __device__ void operator*=(const float4p b) { x *= b.x; y *= b.y; z *= b.z; w *= b.w; }
	inline __host__ __device__ float4p operator*(float b) {	return float4p(x * b,y * b,z * b,w * b); }
	inline __host__ __device__ void operator*=(const float b) { x *= b; y *= b; z *= b; w *= b; }
	inline __host__ __device__ float4p operator/(const float4p b) { return float4p(x / b.x, y / b.y, z / b.z, w / b.w);	}
	inline __host__ __device__ void operator/=(const float4p b)	{ x /= b.x; y /= b.y; z /= b.z; w /= b.w; }
	inline __host__ __device__ float4p operator/(const float b)	{ return float4p(x / b,y / b, z / b, w / b); }
	inline __host__ __device__ void operator/=(const float b) { x /= b; y /= b; z /= b; w /= b;	}
	
};


inline __host__ __device__ float4p operator+(const float4p &a,const float4p &b)
{
	return float4p(a.x + b.x,a.y + b.x,a.z + b.z,a.w + b.w);
}
inline __host__ __device__ float4p operator+(float &b,float4p &a)
{
	return float4p(a.x + b,a.y + b,a.z + b,a.w + b);
}
inline __host__ __device__ float4p operator+(float4p &a,float &b)
{
	return float4p(a.x + b,a.y + b,a.z + b,a.w + b);
}


inline __host__ __device__ float4p operator-(float4p &a)
{
	return float4p(-a.x, -a.y, -a.z , -a.w);
}
inline __host__ __device__ float4p operator-(float4p &a,float4p &b)
{
	return float4p(a.x - b.x,a.y - b.x,a.z - b.z,a.w - b.w);
}
inline __host__ __device__ float4p operator-(float &b,float4p &a)
{
	return float4p(a.x - b,a.y - b,a.z - b,a.w - b);
}
inline __host__ __device__ float4p operator-(float4p &a,float &b)
{
	return float4p(a.x - b,a.y - b,a.z - b,a.w - b);
}

inline __host__ __device__ float4p operator*(float b,float4p a)
{
	return float4p(b * a.x,b * a.y,b * a.z,b * a.w);
}

inline __host__ __device__ float4p operator/(float b,float4p a)
{
	return float4p(b / a.x,b / a.y,b / a.z,b / a.w);
}


inline __host__ __device__ float4p fmaxf(float4p a,float4p b)
{
	return float4p(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z),fmaxf(a.w,b.w));
}
inline __device__ __host__ float4p lerp(float4p a,float4p b,float t)
{
	return a + t*(b-a);
}
inline __device__ __host__ float4p clamp(float4p v,float a,float b)
{
	return float4p(clamp(v.x,a,b),clamp(v.y,a,b),clamp(v.z,a,b),clamp(v.w,a,b));
}
inline __device__ __host__ float4p clamp(float4p v,float4p a,float4p b)
{
	return float4p(clamp(v.x,a.x,b.x),clamp(v.y,a.y,b.y),clamp(v.z,a.z,b.z),clamp(v.w,a.w,b.w));
}
inline __host__ __device__ float dot(float4p a,float4p b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ float dotp(float4p a,float4p b)  // NB this version omits 4th component
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float length(float4p v)
{
	return sqrtf(dot(v,v));
}

inline __host__ __device__ float lengthp(float4p v)
{
	return sqrtf(dotp(v,v));
}
inline __host__ __device__ float4p normalize(float4p v)
{
	float invLen = rsqrtf(dot(v,v));
	return v * invLen;
}
inline __host__ __device__ float4p normalizep(float4p v)
{
	float invLen = rsqrtf(dotp(v,v));
	return v * invLen;
}

inline __host__ __device__ float4p floorf(float4p v)
{
	return float4p(floorf(v.x),floorf(v.y),floorf(v.z),floorf(v.w));
}
inline __host__ __device__ float4p fracf(float4p v)
{
	return float4p(fracf(v.x),fracf(v.y),fracf(v.z),fracf(v.w));
}
inline __host__ __device__ float4p fmodf(float4p a,float4p b)
{
	return float4p(fmodf(a.x,b.x),fmodf(a.y,b.y),fmodf(a.z,b.z),fmodf(a.w,b.w));
}

// NB float3 here
inline __host__ __device__ float4p reflect(float4p i,float4p n)
{
	return i - 2.0f * n * dotp(n,i);
}
inline __host__ __device__ float4p cross(float4p a,float4p b)
{
	return float4p(a.y*b.z - a.z*b.y,a.z*b.x - a.x*b.z,a.x*b.y - a.y*b.x, 0.0f);
}
inline __device__ __host__ float4p smoothstep(float4p a,float4p b,float4p x)
{
	float4p y = clamp((x - a) / (b - a),0.0f,1.0f);
	return (y*y*(float4p(3.0f) - (float4p(2.0f)*y)));
}

#endif

