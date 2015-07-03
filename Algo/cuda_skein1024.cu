#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <stdio.h>
#include <memory.h>
//#include <stdint.h>


typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#include "cuda_helper.h"

extern int device_major[8];
extern int device_minor[8];

// aus cpu-miner.c
extern int device_map[8];
// aus heavy.cu


#define ROL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))


static __constant__ uint2 uMessage[10];
static __constant__ uint2 c_hv[17]; 
static __constant__ uint2 skein_ks_parity = {0x55555555,0x55555555};

static __constant__ int ROT1024[8][8];
static __constant__ uint2 t12[9] = 
{ {0x80, 0}, 
  {0, 0x70000000}, 
  {0x80, 0x70000000},
  {0xd8, 0}, 
  {0, 0xb0000000},
  {0xd8, 0xb0000000},
  {0x08, 0},
  {0, 0xff000000}, 
  {0x08, 0xff000000}
};


static const uint64_t cpu_SKEIN1024_IV_1024[16] =
{
	//     lo           hi
	 0x5A4352BE62092156,
	 0x5F6E8B1A72F001CA,
	 0xFFCBFE9CA1A2CE26,
	 0x6C23C39667038BCA,
	 0x583A8BFCCE34EB6C,
	 0x3FDBFB11D4A46A3E, 
	 0x3304ACFCA8300998, 
	 0xB2F6675FA17F0FD2,
	 0x9D2599730EF7AB6B,
	 0x0914A20D3DFEA9E4, 
	 0xCC1A9CAFA494DBD3,
	 0x9828030DA0A6388C,
	 0x0D339D5DAADEE3DC, 
	 0xFC46DE35C4E2A086, 
	 0x53D6E4F52E19A6D1,
	 0x5663952F715D1DDD, 
};
static const int cpu_ROT1024[8][8] =
{
	{ 55, 43, 37, 40, 16, 22, 38, 12 },
	{ 25, 25, 46, 13, 14, 13, 52, 57 },
	{ 33, 8, 18, 57, 21, 12, 32, 54 },
	{ 34, 43, 25, 60, 44, 9, 59, 34 },
	{ 28, 7, 47, 48, 51, 9, 35, 41 },
	{ 17, 6, 18, 25, 43, 42, 40, 15 },
	{ 58, 7, 32, 45, 19, 18, 2, 56 },
	{ 47, 49, 27, 58, 37, 48, 53, 56 }
};
//
/*
static __forceinline__ __device__ void Round1024(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, 
                                                 uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
p0 += p1; 
p1 = ROL2(p1, ROT1024[ROT][0]); 
p1 ^= p0;   
p2 += p3; 
p3 = ROL2(p3, ROT1024[ROT][1]); 
p3 ^= p2;   
p4 += p5; 
p5 = ROL2(p5, ROT1024[ROT][2]); 
p5 ^= p4;   
p6 += p7; 
p7 = ROL2(p7, ROT1024[ROT][3]); 
p7 ^= p6;   
p8 += p9; 
p9 = ROL2(p9, ROT1024[ROT][4]); 
p9 ^= p8;   
pA += pB; 
pB = ROL2(pB, ROT1024[ROT][5]); 
pB ^= pA;   
pC += pD; 
pD = ROL2(pD, ROT1024[ROT][6]); 
pD ^= pC;   
pE += pF; 
pF = ROL2(pF, ROT1024[ROT][7]); 
pF ^= pE;
}

static __forceinline__ __device__ void Round1024_v2(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	uint2 &p8, uint2 &p9, uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF, int ROT)
{
	p0 += p1;
    p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;
	p1 = ROL2(p1, ROT1024[ROT][0]);	
	p3 = ROL2(p3, ROT1024[ROT][1]);
	p5 = ROL2(p5, ROT1024[ROT][2]);
	p7 = ROL2(p7, ROT1024[ROT][3]);
	p9 = ROL2(p9, ROT1024[ROT][4]);
	pB = ROL2(pB, ROT1024[ROT][5]);
	pD = ROL2(pD, ROT1024[ROT][6]);
	pF = ROL2(pF, ROT1024[ROT][7]);
	p1 ^= p0;
	p3 ^= p2;
	p5 ^= p4;
	p7 ^= p6;
	p9 ^= p8;
	pB ^= pA;
	pD ^= pC;
	pF ^= pE;
}
*/
#define Round1024(p0, p1, p2, p3, p4, p5, p6,p7,p8, p9, pA, pB, pC, pD, pE, pF, ROT) \
{ \
	p0 += p1; \
	p2 += p3; \
	p4 += p5; \
	p6 += p7; \
	p8 += p9; \
	pA += pB; \
	pC += pD; \
	pE += pF; \
	p1 = ROL2(p1, ROT1024[ROT][0]); \
	p3 = ROL2(p3, ROT1024[ROT][1]); \
	p5 = ROL2(p5, ROT1024[ROT][2]); \
	p7 = ROL2(p7, ROT1024[ROT][3]); \
	p9 = ROL2(p9, ROT1024[ROT][4]); \
	pB = ROL2(pB, ROT1024[ROT][5]); \
	pD = ROL2(pD, ROT1024[ROT][6]); \
	pF = ROL2(pF, ROT1024[ROT][7]); \
	p1 ^= p0; \
	p3 ^= p2; \
	p5 ^= p4; \
	p7 ^= p6; \
	p9 ^= p8; \
	pB ^= pA; \
	pD ^= pC; \
	pF ^= pE; \
}


static __forceinline__ __host__ void Round1024_host(uint64_t &p0, uint64_t &p1, uint64_t &p2, uint64_t &p3, uint64_t &p4, uint64_t &p5, uint64_t &p6, uint64_t &p7,
	uint64_t &p8, uint64_t &p9, uint64_t &pA, uint64_t &pB, uint64_t &pC, uint64_t &pD, uint64_t &pE, uint64_t &pF, int ROT)
{
	p0 += p1;
	p1 = ROL64(p1, cpu_ROT1024[ROT][0]);
	p1 ^= p0;
	p2 += p3;
	p3 = ROL64(p3, cpu_ROT1024[ROT][1]);
	p3 ^= p2;
	p4 += p5;
	p5 = ROL64(p5, cpu_ROT1024[ROT][2]);
	p5 ^= p4;
	p6 += p7;
	p7 = ROL64(p7, cpu_ROT1024[ROT][3]);
	p7 ^= p6;
	p8 += p9;
	p9 = ROL64(p9, cpu_ROT1024[ROT][4]);
	p9 ^= p8;
	pA += pB;
	pB = ROL64(pB, cpu_ROT1024[ROT][5]);
	pB ^= pA;
	pC += pD;
	pD = ROL64(pD, cpu_ROT1024[ROT][6]);
	pD ^= pC;
	pE += pF;
	pF = ROL64(pF, cpu_ROT1024[ROT][7]);
	pF ^= pE;
}

//original 256,2
__global__  __launch_bounds__(256, 3) void  skein1024_gpu_hash_35(int threads, uint64_t startNonce, uint2 *outputHash)
{
    


	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
//	{
		// Skein
		uint2 h[17];
		
        uint2 t[3];
		uint2 p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,p14, p15;
		uint64_t nonce = startNonce + (uint64_t)thread;

		p0 = uMessage[0];    
		p1 = uMessage[1];
		p2 = uMessage[2];
		p3 = uMessage[3];
		p4 = uMessage[4];
		p5 = uMessage[5];
		p6 = uMessage[6];
		p7 = uMessage[7];
		p8 = uMessage[8];    
		p9 = uMessage[9];    
		p10 = vectorize(nonce); 
				
		uint2 tempnonce = p10;
		t[0] = t12[3]; // ptr  
		t[1] = t12[4]; // etype
		t[2] = t12[5];
		
		p0  += c_hv[0];
		p1  += c_hv[1];
		p2  += c_hv[2];
		p3  += c_hv[3];
		p4  += c_hv[4];
		p5  += c_hv[5];
		p6  += c_hv[6];
		p7  += c_hv[7];
		p8  += c_hv[8];
		p9  += c_hv[9];
		p10 += c_hv[10];
		p11  = c_hv[11];
		p12  = c_hv[12];
		p13  = c_hv[13] +t[0];
		p14  = c_hv[14] +t[1];
		p15  = c_hv[15];
#pragma unroll
		for (int i = 1; i < 21; i++)
		{
			uint32_t truc = 4 * ((i-1)&1);
			Round1024(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 0+truc);
			Round1024(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 1+truc);
			Round1024(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 2+truc);
			Round1024(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 3+truc);

			p0  += c_hv[(i + 0) % 17];
			p1  += c_hv[(i + 1) % 17];
			p2  += c_hv[(i + 2) % 17];
			p3  += c_hv[(i + 3) % 17];
			p4  += c_hv[(i + 4) % 17];
			p5  += c_hv[(i + 5) % 17];
			p6  += c_hv[(i + 6) % 17];
			p7  += c_hv[(i + 7) % 17];
			p8  += c_hv[(i + 8) % 17];
			p9  += c_hv[(i + 9) % 17];
			p10 += c_hv[(i + 10) % 17];
			p11 += c_hv[(i + 11) % 17];
			p12 += c_hv[(i + 12) % 17];
			p13 += c_hv[(i + 13) % 17] + t[(i + 0) % 3];
			p14 += c_hv[(i + 14) % 17] + t[(i + 1) % 3];
			p15 += c_hv[(i + 15) % 17] + make_uint2(i, 0);			
}


		p0  ^= uMessage[0];
		p1  ^= uMessage[1];
		p2  ^= uMessage[2];
		p3  ^= uMessage[3];
		p4  ^= uMessage[4];
		p5  ^= uMessage[5];
		p6  ^= uMessage[6];
		p7  ^= uMessage[7];
		p8  ^= uMessage[8];
		p9  ^= uMessage[9];
		p10 ^= tempnonce;

////////////////////////////// round 3 /////////////////////////////////////
		h[0]  = p0;
		h[1]  = p1;
		h[2]  = p2;
		h[3]  = p3;
		h[4]  = p4;
		h[5]  = p5;
		h[6]  = p6;
		h[7]  = p7;
		h[8]  = p8;
		h[9]  = p9;
		h[10] = p10;
		h[11] = p11;
		h[12] = p12;
		h[13] = p13;
		h[14] = p14;
		h[15] = p15;
		h[16] = skein_ks_parity;
        #pragma unroll 16
		for (int i = 0; i<16; i++) {h[16] ^= h[i];}

		t[0] = t12[6]; 
		t[1] = t12[7]; 
		t[2] = t12[8];

		p13 += t[0];  //p13 already equal h[13] 
		p14 += t[1];
      #pragma unroll
		for (int i = 1; i < 21; i++)
		{
			uint32_t truc = 4 * ((i - 1) & 1);
			Round1024(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 0+truc);
			Round1024(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 1+truc);
			Round1024(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 2+truc);
			Round1024(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 3+truc);

			p0  += h[(i + 0) % 17];
			p1  += h[(i + 1) % 17];
			p2  += h[(i + 2) % 17];
			p3  += h[(i + 3) % 17];
			p4  += h[(i + 4) % 17];
			p5  += h[(i + 5) % 17];
			p6  += h[(i + 6) % 17];
			p7  += h[(i + 7) % 17];
			p8  += h[(i + 8) % 17];
			p9  += h[(i + 9) % 17];
			p10 += h[(i + 10) % 17];
			p11 += h[(i + 11) % 17];
			p12 += h[(i + 12) % 17];
			p13 += h[(i + 13) % 17] + t[(i + 0) % 3];
			p14 += h[(i + 14) % 17] + t[(i + 1) % 3];
			p15 += h[(i + 15) % 17] + make_uint2(i, 0);			
}

			outputHash[               thread] = p0;
		    outputHash[1 *2*  threads + thread] = p1;
			outputHash[2 * 2 * threads + thread] = p2;
			outputHash[3 * 2 * threads + thread] = p3;
			outputHash[4 * 2 * threads + thread] = p4;
			outputHash[5 * 2 * threads + thread] = p5;
			outputHash[6 * 2 * threads + thread] = p6;
			outputHash[7 * 2 * threads + thread] = p7;
			outputHash[8 * 2 * threads + thread] = p8;
			outputHash[9 * 2 * threads + thread] = p9;
			outputHash[10 * 2 * threads + thread] = p10;
			outputHash[11 * 2 * threads + thread] = p11;
			outputHash[12 * 2 * threads + thread] = p12;
			outputHash[13 * 2 * threads + thread] = p13;
			outputHash[14 * 2 * threads + thread] = p14;
			outputHash[15 * 2 * threads + thread] = p15;

//	} // thread

}




__host__ void skein1024_cpu_init(int thr_id, int threads)
{
	
	cudaMemcpyToSymbol(ROT1024, cpu_ROT1024, sizeof(cpu_ROT1024), 0, cudaMemcpyHostToDevice);
}

__host__ void skein1024_cpu_hash(int thr_id, int threads, uint64_t startNounce, uint2 *d_outputHash, int order)
{
 
	int threadsperblock = 256; 
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

		skein1024_gpu_hash_35 << <grid, block>> >(threads, startNounce, d_outputHash);

//	MyStreamSynchronize(NULL, order, thr_id);
	
}

__host__ void skein1024_setBlock(void *pdata)
{

	uint2 hv[17];
	uint64_t t[3];
	uint64_t h[17];
    uint64_t p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15;
	
	uint64_t cpu_skein_ks_parity = 0x5555555555555555;
	h[16] = cpu_skein_ks_parity;
	for (int i = 0; i<16; i++) {
		h[i] = cpu_SKEIN1024_IV_1024[i];
		h[16] ^= h[i];}
	uint64_t* alt_data = (uint64_t*)pdata;
	/////////////////////// round 1 //////////////////////////// should be on cpu => constant on gpu
	p0 = alt_data[0];
	p1 = alt_data[1];
	p2 = alt_data[2];
	p3 = alt_data[3];
	p4 = alt_data[4];
	p5 = alt_data[5];
	p6 = alt_data[6];
	p7 = alt_data[7];
	p8 = alt_data[8];
	p9 = alt_data[9];
	p10 = alt_data[10];
	p11 = alt_data[11];
	p12 = alt_data[12];
	p13 = alt_data[13];
	p14 = alt_data[14];
	p15 = alt_data[15];
	t[0] = 0x80; // ptr  
	t[1] = 0x7000000000000000; // etype
	t[2] = 0x7000000000000080;

	p0 += h[0];
	p1 += h[1];
	p2 += h[2];
	p3 += h[3];
	p4 += h[4];
	p5 += h[5];
	p6 += h[6];
	p7 += h[7];
	p8 += h[8];
	p9 += h[9];
	p10 += h[10];
	p11 += h[11];
	p12 += h[12];
	p13 += h[13] + t[0];
	p14 += h[14] + t[1];
	p15 += h[15];

	for (int i = 1; i < 21; i += 2)
	{

		Round1024_host(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 0);
		Round1024_host(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 1);
		Round1024_host(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 2);
		Round1024_host(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 3);

		p0 += h[(i + 0) % 17];
		p1 += h[(i + 1) % 17];
		p2 += h[(i + 2) % 17];
		p3 += h[(i + 3) % 17];
		p4 += h[(i + 4) % 17];
		p5 += h[(i + 5) % 17];
		p6 += h[(i + 6) % 17];
		p7 += h[(i + 7) % 17];
		p8 += h[(i + 8) % 17];
		p9 += h[(i + 9) % 17];
		p10 += h[(i + 10) % 17];
		p11 += h[(i + 11) % 17];
		p12 += h[(i + 12) % 17];
		p13 += h[(i + 13) % 17] + t[(i + 0) % 3];
		p14 += h[(i + 14) % 17] + t[(i + 1) % 3];
		p15 += h[(i + 15) % 17] + (uint64_t)i;

		Round1024_host(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 4);
		Round1024_host(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 5);
		Round1024_host(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 6);
		Round1024_host(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 7);

		p0 += h[(i + 1) % 17];
		p1 += h[(i + 2) % 17];
		p2 += h[(i + 3) % 17];
		p3 += h[(i + 4) % 17];
		p4 += h[(i + 5) % 17];
		p5 += h[(i + 6) % 17];
		p6 += h[(i + 7) % 17];
		p7 += h[(i + 8) % 17];
		p8 += h[(i + 9) % 17];
		p9 += h[(i + 10) % 17];
		p10 += h[(i + 11) % 17];
		p11 += h[(i + 12) % 17];
		p12 += h[(i + 13) % 17];
		p13 += h[(i + 14) % 17] + t[(i + 1) % 3];
		p14 += h[(i + 15) % 17] + t[(i + 2) % 3];
		p15 += h[(i + 16) % 17] + (uint64_t)(i + 1);


	}

	h[0] = p0^alt_data[0];
	h[1] = p1^alt_data[1];	
    h[2] = p2^alt_data[2];
	h[3] = p3^alt_data[3];
	h[4] = p4^alt_data[4];
	h[5] = p5^alt_data[5];
	h[6] = p6^alt_data[6];
	h[7] = p7^alt_data[7];
	h[8] = p8^alt_data[8];
	h[9] = p9^alt_data[9];
	h[10] = p10^alt_data[10];
	h[11] = p11^alt_data[11];
	h[12] = p12^alt_data[12];
	h[13] = p13^alt_data[13];
	h[14] = p14^alt_data[14];
	h[15] = p15^alt_data[15];
	h[16] = cpu_skein_ks_parity;
	for (int i = 0; i<16; i++) { h[16] ^= h[i]; }
	for (int i = 0; i<17; i++) { hv[i] = lohi_host(h[i]); } //will slow down things


	uint2 cpu_Message[10];
	for (int i = 0; i<10; i++) { cpu_Message[i] = lohi_host(alt_data[i+16]);} //might slow down things
   
	cudaMemcpyToSymbol( c_hv, hv, sizeof(hv), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(uMessage, cpu_Message, sizeof(cpu_Message), 0, cudaMemcpyHostToDevice);
	

}