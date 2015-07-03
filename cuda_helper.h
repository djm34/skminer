#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#ifdef __INTELLISENSE__
#define __launch_bounds__(x)
#endif


static __device__ unsigned long long MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
uint64_t result;
asm volatile("{\n\t"
	"mov.b64 %0,{%1,%2}; \n\t"
		"}"
		: "=l"(result) : "r"(LO) , "r"(HI));
return result;
}
static __device__ uint32_t HIWORD(uint64_t x)
{
uint32_t result;
asm ("{\n\t"
	".reg .u32 xl; \n\t"
	"mov.b64 {xl,%0},%1; \n\t"
		"}"
		: "=r"(result) : "l"(x));
return result;
}

static __device__ void LOHI(uint32_t &lo, uint32_t &hi,uint64_t x)
{
	asm("{\n\t"
		"mov.b64 {%0,%1},%2; \n\t"
		"}"
		: "=r"(lo), "=r"(hi) : "l"(x));
}

static __device__ uint32_t LOWORD(uint64_t x)
{
uint32_t result;
asm ("{\n\t"
	".reg .u32 xh; \n\t"
	"mov.b64 {%0,xh},%1; \n\t"
		"}"
		: "=r"(result) : "l"(x));
return result;
}
// das Hi Word aus einem 64 Bit Typen extrahieren


#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define SPH_ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
    #define SPH_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
    // Kepler (Compute 3.5)
    #define SPH_ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
    #define SPH_ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif

// das Hi Word in einem 64 Bit Typen ersetzen
static __device__ uint64_t oREPLACE_HIWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32ULL);
}

static __host__ uint2 lohi_host(const uint64_t x)
{ uint2 res;
res.x = (uint32_t)(x & 0xFFFFFFFFULL);
res.y = (uint32_t)(x >> 32);
return res;
}
static __host__ uint64_t make64_host(const uint2 x)
{
	return (uint64_t)x.x | (((uint64_t)x.y) << 32);
}

static __device__ uint64_t REPLACE_HIWORD(uint64_t x, uint32_t y) {
	asm ("{\n\t"
		" .reg .u32 tl,th; \n\t"
		"mov.b64 {tl,th},%0; \n\t"
		"mov.b64 %0,{tl,%1}; \n\t"
		"}"
		: "+l"(x) : "r"(y) );
return x;
}


static __device__ uint64_t REPLACE_LOWORD(uint64_t x, uint32_t y) {
        asm ("{\n\t"
		" .reg .u32 tl,th; \n\t"
		"mov.b64 {tl,th},%0; \n\t"
		"mov.b64 %0,{%1,th}; \n\t"
		"}"
		: "+l"(x) : "r"(y) );
return x;
}






__forceinline__ __device__ uint64_t sph_t64(uint64_t x)
{
uint64_t result;
 asm("{\n\t"
    "and.b64 %0,%1,0xFFFFFFFFFFFFFFFF;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x));
	return result;
}
__forceinline__ __device__ uint32_t sph_t32(uint32_t x)
{
uint32_t result;
 asm("{\n\t"
    "and.b32 %0,%1,0xFFFFFFFF;\n\t"
    "}\n\t"
	: "=r"(result) : "r"(x));
	return result;
}


__forceinline__ __device__ uint64_t shr_t64(uint64_t x,uint32_t n)
{
uint64_t result;
asm("{\n\t"
	"shr.b64 %0,%1,%2;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}
__forceinline__ __device__ uint64_t shl_t64(uint64_t x,uint32_t n)
{
uint64_t result;
asm("{\n\t"
	"shl.b64 %0,%1,%2;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}
__forceinline__ __device__ uint32_t shr_t32(uint32_t x,uint32_t n)
{
uint32_t result;
asm("{\n\t"
	"shr.b32 %0,%1,%2;\n\t"
    "}\n\t"
	: "=r"(result) : "r"(x), "r"(n));
	return result;
}
__forceinline__ __device__ uint32_t shl_t32(uint32_t x,uint32_t n)
{
uint32_t result;
asm("{\n\t"
	"shl.b32 %0,%1,%2;\n\t"
    "}\n\t"
	: "=r"(result) : "r"(x), "r"(n));
	return result;
}


__forceinline__ __device__ uint64_t mul(uint64_t a,uint64_t b)
{
uint64_t result;
asm("{\n\t"
	"mul.lo.u64 %0,%1,%2; \n\t"    
     "}\n\t"
	: "=l"(result) : "l"(a) , "l"(b));
return result;
}

///uint2 method

#if  __CUDA_ARCH__ >= 350 
__inline__ __device__ uint2 ROR2(const uint2 a, const int offset) {
	uint2 result;
	if (offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}
#else
__inline__ __device__ uint2 ROR2(const uint2 v, const int n) {
	uint2 result;
    result.x = (((v.x) >> (n)) | ((v.x) << (64 - (n))));
	result.y = (((v.y) >> (n)) | ((v.y) << (64 - (n))));
	return result;
}
#endif


#if  __CUDA_ARCH__ >= 350 
__inline__ __device__ uint2 ROL2(const uint2 a, const int offset) {
	uint2 result;
	if (offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}
#else
__inline__ __device__ uint2 ROL2(const uint2 v, const int n) {
	uint2 result;
	result.x = (((v.x) << (n)) | ((v.x) >> (64 - (n))));
	result.y = (((v.y) << (n)) | ((v.y) >> (64 - (n))));
	return result;
}
#endif

static __forceinline__ __device__ uint64_t devectorize(uint2 v) { return MAKE_ULONGLONG(v.x, v.y); }
static __forceinline__ __device__ uint2 vectorize(uint64_t v) {
	uint2 result;
	LOHI(result.x, result.y, v);
	return result;
}

static __forceinline__ __device__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
static __forceinline__ __device__ uint4 operator^ (uint4 a, uint4 b) { return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w ); }
static __forceinline__ __device__ uint2 operator& (uint2 a, uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }
static __forceinline__ __device__ uint2 operator| (uint2 a, uint2 b) { return make_uint2(a.x | b.x, a.y | b.y); }
static __forceinline__ __device__ uint2 operator~ (uint2 a) { return make_uint2(~a.x, ~a.y); }
static __forceinline__ __device__  void operator^= (uint2 &a, uint2 b) { a = a ^ b; }
static __forceinline__ __device__  void operator^= (uint4 &a, uint4 b) { a = a ^ b; }
static __forceinline__ __device__ uint2 operator+ (uint2 a, uint2 b) 
{
uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x),"=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x),"r"(b.y));
return result;
}
static __forceinline__ __device__ void operator+= (uint2 &a, uint2 b) {a = a + b;}

static __forceinline__ __device__ uint2 operator* (uint2 a, uint2 b)
{ //basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b)) 
//(what does uint64 "*" operator) 
	uint2 result;
	asm("{\n\t"
		"mul.lo.u32        %0,%2,%4;  \n\t"
		"mul.hi.u32        %1,%2,%4;  \n\t"
		"mad.lo.cc.u32    %1,%3,%4,%1; \n\t"
		"madc.lo.u32      %1,%3,%5,%1; \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}
#if  __CUDA_ARCH__ >= 350 
static __forceinline__ __device__ uint2 shiftl2 (uint2 a, int offset) 
{
uint2 result;
if (offset<32) {
	asm("{\n\t"
		"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
		"shl.b32 %0,%2,%4; \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
} else {
	asm("{\n\t"
		"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
		"shl.b32 %0,%2,%4; \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
}
return result;
}
static __forceinline__ __device__ uint2 shiftr2(uint2 a, int offset)
{
	uint2 result;
	if (offset<32) {
		asm("{\n\t"
			"shf.r.clamp.b32 %0,%2,%3,%4; \n\t"
			"shr.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else {
		asm("{\n\t"
			"shf.l.clamp.b32 %0,%2,%3,%4; \n\t"
			"shl.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
}
#else 
static __forceinline__ __device__ uint2 shiftl2(uint2 a, int offset)
{
uint2 result;
	asm("{\n\t"
		".reg .b64 u,v; \n\t"
		"mov.b64 v,{%2,%3}; \n\t"
		"shl.b64 u,v,%4; \n\t"
		"mov.b64 {%0,%1},v;  \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
return result;
}
static __forceinline__ __device__ uint2 shiftr2(uint2 a, int offset)
{
	uint2 result;
	asm("{\n\t"
		".reg .b64 u,v; \n\t"
		"mov.b64 v,{%2,%3}; \n\t"
		"shr.b64 u,v,%4; \n\t"
		"mov.b64 {%0,%1},v;  \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return result;
}
#endif

__device__ __inline__ uint4 make_uint4f2(uint2 a, uint2 b)
{
	return (make_uint4(a.x, a.y, b.x, b.y));

}


typedef struct __align__(16) uint42
{
	union {
		struct { uint2  x, y; };
		uint4 to4;
	};
} uint42;

static __inline__ __host__ __device__ uint42 make_uint42(const uint2 &a, const uint2 &b)
{
	uint42 t; t.x = a; t.y = b; return t;
}
static __inline__ __host__ __device__ uint42 make_uint42(const uint4 &a)
{
	uint42 t; t.to4 = a; return t;
}


#endif // #ifndef CUDA_HELPER_H
