
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>


//extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
extern int device_major[8];
extern int device_minor[8];

__constant__ uint64_t pTarget[16];
#include "cuda_helper.h"

uint64_t *d_SKNonce[8];

#define ROL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))

static __constant__ uint64_t RC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull,
    0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull,
    0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull,
    0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull,
    0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull,
    0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull,
    0x0000000080000001ull, 0x8000000080008008ull
};


static __device__ __forceinline__ void keccak_block(uint2  *s, const uint64_t *keccak_round_constants) {
	size_t i;
	uint2 v, w;
	uint2 t[5];
    uint2 u[5];
//    #pragma unroll
	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
//		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
//		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
//		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
//		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
//		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];
				t[0] = s[0] ^ s[5]; 
				t[1] = s[1] ^ s[6]; 
				t[2] = s[2] ^ s[7]; 
				t[3] = s[3] ^ s[8];
				t[4] = s[4] ^ s[9]; 
				t[0] ^= s[10]; 
				t[1] ^= s[11]; 
				t[2] ^= s[12]; 
				t[3] ^= s[13]; 
				t[4] ^= s[14]; 
				t[0] ^= s[15];
				t[1] ^= s[16];
				t[2] ^= s[17];
				t[3] ^= s[18];
				t[4] ^= s[19];
				t[0] ^= s[20];
				t[1] ^= s[21];
				t[2] ^= s[22];
				t[3] ^= s[23];
				t[4] ^= s[24];
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		
		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROL2(s[6], 44);
		s[6] = ROL2(s[9], 20);
		s[9] = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2] = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5] = ROL2(s[3], 28);
		s[3] = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7] = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= vectorize(keccak_round_constants[i]);
	}
} 

static __device__ __forceinline__ void keccak_blocklast(uint2  *s, const uint64_t *keccak_round_constants,uint64_t &comp) {
	size_t i;
	uint2 v, w;
	uint2 t[5];
	uint2 u[5];
	//    #pragma unroll
	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		//		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		//		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		//		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		//		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		//		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];
		t[0] = s[0] ^ s[5];
		t[1] = s[1] ^ s[6];
		t[2] = s[2] ^ s[7];
		t[3] = s[3] ^ s[8];
		t[4] = s[4] ^ s[9];
		t[0] ^= s[10];
		t[1] ^= s[11];
		t[2] ^= s[12];
		t[3] ^= s[13];
		t[4] ^= s[14];
		t[0] ^= s[15];
		t[1] ^= s[16];
		t[2] ^= s[17];
		t[3] ^= s[18];
		t[4] ^= s[19];
		t[0] ^= s[20];
		t[1] ^= s[21];
		t[2] ^= s[22];
		t[3] ^= s[23];
		t[4] ^= s[24];
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];


		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROL2(s[6], 44);
		s[6] = ROL2(s[9], 20);
		s[9] = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2] = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5] = ROL2(s[3], 28);
		s[3] = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7] = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= vectorize(keccak_round_constants[i]);
	}
	comp = devectorize(s[6]);
}

static __device__ __forceinline__ void keccak_block1st(uint2  *s, const uint64_t *keccak_round_constants) {
	size_t i;
	uint2 v, w;
	uint2 t[5];
	uint2 u[5];
	//    #pragma unroll
	t[0] = s[0] ^ s[5];
	t[1] = s[1] ^ s[6];
	t[2] = s[2] ^ s[7];
	t[3] = s[3] ^ s[8];
	t[4] = s[4];
	
	/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
	u[0] = t[4] ^ ROL2(t[1], 1);
	u[1] = t[0] ^ ROL2(t[2], 1);
	u[2] = t[1] ^ ROL2(t[3], 1);
	u[3] = t[2] ^ ROL2(t[4], 1);
	u[4] = t[3] ^ ROL2(t[0], 1);

	/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

	s[0] ^= u[0]; s[5] ^= u[0]; s[10] = u[0]; s[15] = u[0]; s[20] = u[0];
	s[1] ^= u[1]; s[6] ^= u[1]; s[11] = u[1]; s[16] = u[1]; s[21] = u[1];
	s[2] ^= u[2]; s[7] ^= u[2]; s[12] = u[2]; s[17] = u[2]; s[22] = u[2];
	s[3] ^= u[3]; s[8] ^= u[3]; s[13] = u[3]; s[18] = u[3]; s[23] = u[3];
	s[4] ^= u[4]; s[9]  = u[4]; s[14] = u[4]; s[19] = u[4]; s[24] = u[4];


	/* rho pi: b[..] = rotl(a[..], ..) */
	v = s[1];
	s[1] = ROL2(s[6], 44);
	s[6] = ROL2(s[9], 20);
	s[9] = ROL2(s[22], 61);
	s[22] = ROL2(s[14], 39);
	s[14] = ROL2(s[20], 18);
	s[20] = ROL2(s[2], 62);
	s[2] = ROL2(s[12], 43);
	s[12] = ROL2(s[13], 25);
	s[13] = ROL2(s[19], 8);
	s[19] = ROL2(s[23], 56);
	s[23] = ROL2(s[15], 41);
	s[15] = ROL2(s[4], 27);
	s[4] = ROL2(s[24], 14);
	s[24] = ROL2(s[21], 2);
	s[21] = ROL2(s[8], 55);
	s[8] = ROL2(s[16], 45);
	s[16] = ROL2(s[5], 36);
	s[5] = ROL2(s[3], 28);
	s[3] = ROL2(s[18], 21);
	s[18] = ROL2(s[17], 15);
	s[17] = ROL2(s[11], 10);
	s[11] = ROL2(s[7], 6);
	s[7] = ROL2(s[10], 3);
	s[10] = ROL2(v, 1);

	/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

	v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
	v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
	v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
	v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
	v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

	/* iota: a[0,0] ^= round constant */
	s[0] ^= vectorize(keccak_round_constants[0]);
	for (i = 1; i < 24; i++) {

		t[0] = s[0] ^ s[5];
		t[1] = s[1] ^ s[6];
		t[2] = s[2] ^ s[7];
		t[3] = s[3] ^ s[8];
		t[4] = s[4] ^ s[9];
		t[0] ^= s[10];
		t[1] ^= s[11];
		t[2] ^= s[12];
		t[3] ^= s[13];
		t[4] ^= s[14];
		t[0] ^= s[15];
		t[1] ^= s[16];
		t[2] ^= s[17];
		t[3] ^= s[18];
		t[4] ^= s[19];
		t[0] ^= s[20];
		t[1] ^= s[21];
		t[2] ^= s[22];
		t[3] ^= s[23];
		t[4] ^= s[24];
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];


		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROL2(s[6], 44);
		s[6] = ROL2(s[9], 20);
		s[9] = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2] = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5] = ROL2(s[3], 28);
		s[3] = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7] = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= vectorize(keccak_round_constants[i]);
	}

}


__global__ __launch_bounds__(256,3) void  sk1024_keccak_gpu_hash(int threads, uint64_t startNonce, uint2 *g_hash, uint64_t *resNounce)
{

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
//    if (thread < threads)
//    {
        
		uint64_t nonce = startNonce + thread;

         uint2 state[25];
		 uint2 hash[15];
         uint64_t comp;
         
       #pragma unroll 9
	   for (int i = 0; i<9; i++) {state[i]=g_hash[i * 2 * threads + thread];}
       #pragma unroll 16
	   for (int i = 9; i<25; i++) { state[i] = make_uint2(0,0); }
	   keccak_block(state,RC);
       #pragma unroll 7
	   for (int i = 0; i<7; i++) { state[i] ^= g_hash[(9+i) * 2 * threads + thread]; }
	   state[7].x ^= 0x05;
	   state[8].y ^= 0x80000000; // vectorize(1ULL << 63);
	   keccak_block(state, RC);
	   keccak_blocklast(state, RC,comp);
	   if (comp <= pTarget[15]) { resNounce[0] = nonce; }

//	} //thread
}

__global__ __launch_bounds__(256, 3) void  sk1024_keccak_gpu_hash50(int threads, uint64_t startNonce, uint2 *g_hash,  uint64_t *resNounce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
//	{

		uint64_t nonce = startNonce + thread;

		uint2 state[25];
		uint2 hash[15];
		uint64_t comp;
		
#pragma unroll 9
		for (int i = 0; i<9; i++) { state[i] = g_hash[i * 2 * threads + thread]; }
#pragma unroll 16
		for (int i = 9; i<25; i++) { state[i] = make_uint2(0, 0); }
		keccak_block(state, RC);
#pragma unroll 7
		for (int i = 0; i<7; i++) { state[i] ^= g_hash[(9+i) * 2 * threads + thread]; }
		state[7].x ^= 0x05;
		state[8].y ^= 0x80000000; // vectorize(1ULL << 63);
		
		keccak_block(state, RC);
		keccak_blocklast(state, RC,comp);
		if (comp <= pTarget[15]) { resNounce[0] = nonce; }
//	} //thread

}


   
void sk1024_keccak_cpu_init(int thr_id, int threads)
{
    	
	cudaMalloc(&d_SKNonce[thr_id], sizeof(uint64_t));
	
} 


__host__ uint64_t sk1024_keccak_cpu_hash(int thr_id, int threads, uint64_t startNounce, uint2 *d_hash, int order)
{
	uint64_t result = 0;
    cudaMemset(d_SKNonce[thr_id], 0, sizeof(uint64_t));
	int threadsperblock = 256;
	
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	
		sk1024_keccak_gpu_hash50 << <grid, block >> >(threads, startNounce, d_hash,  d_SKNonce[thr_id]);
	
	cudaMemcpy(&result, d_SKNonce[thr_id], sizeof(uint64_t), cudaMemcpyDeviceToHost);
	//cudaThreadSynchronize();
//	MyStreamSynchronize(NULL, order, thr_id);
	return result;
}
__host__ void sk1024_set_Target(const void *ptarget)
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol(pTarget, ptarget, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}
