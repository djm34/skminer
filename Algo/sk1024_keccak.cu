
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>


__constant__ uint64_t pTarget[16];
#include "cuda_helper.h"
uint64_t *d_sknounce[8];
uint64_t *d_SKNonce[8];

static __constant__ uint64_t RC[24];
static const uint64_t cpu_RC[24] = {
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



static __device__ __forceinline__ void keccak_1600(uint64_t *state, const uint64_t *keccak_round_constants)
{

	uint2 Aba, Abe, Abi, Abo, Abu;
	uint2 Aga, Age, Agi, Ago, Agu;
	uint2 Aka, Ake, Aki, Ako, Aku;
	uint2 Ama, Ame, Ami, Amo, Amu;
	uint2 Asa, Ase, Asi, Aso, Asu;
	uint2 BCa, BCe, BCi, BCo, BCu;
	uint2 Da, De, Di, Do, Du;
	uint2 Eba, Ebe, Ebi, Ebo, Ebu;
	uint2 Ega, Ege, Egi, Ego, Egu;
	uint2 Eka, Eke, Eki, Eko, Eku;
	uint2 Ema, Eme, Emi, Emo, Emu;
	uint2 Esa, Ese, Esi, Eso, Esu;
	Aba = vectorize(state[0]);
	Abe = vectorize(state[1]);
	Abi = vectorize(state[2]);
	Abo = vectorize(state[3]);
	Abu = vectorize(state[4]);
	Aga = vectorize(state[5]);
	Age = vectorize(state[6]);
	Agi = vectorize(state[7]);
	Ago = vectorize(state[8]);
	Agu = vectorize(state[9]);
	Aka = vectorize(state[10]);
	Ake = vectorize(state[11]);
	Aki = vectorize(state[12]);
	Ako = vectorize(state[13]);
	Aku = vectorize(state[14]);
	Ama = vectorize(state[15]);
	Ame = vectorize(state[16]);
	Ami = vectorize(state[17]);
	Amo = vectorize(state[18]);
	Amu = vectorize(state[19]);
	Asa = vectorize(state[20]);
	Ase = vectorize(state[21]);
	Asi = vectorize(state[22]);
	Aso = vectorize(state[23]);
	Asu = vectorize(state[24]);
//    #pragma unroll 
	for (int round = 0; round < 24; round += 2)
	{
		//    int round =2;
		//    prepareTheta
		BCa = Aba^Aga^Aka^Ama^Asa;
		BCe = Abe^Age^Ake^Ame^Ase;
		BCi = Abi^Agi^Aki^Ami^Asi;
		BCo = Abo^Ago^Ako^Amo^Aso;
		BCu = Abu^Agu^Aku^Amu^Asu;

		//thetaRhoPiChiIotaPrepareTheta(round  , A, E)
		Da = BCu^ROL2(BCe, 1);
		De = BCa^ROL2(BCi, 1);
		Di = BCe^ROL2(BCo, 1);
		Do = BCi^ROL2(BCu, 1);
		Du = BCo^ROL2(BCa, 1);

		Aba ^= Da;
		BCa = Aba;
		Age ^= De;
		BCe = ROL2(Age, 44);
		Aki ^= Di;
		BCi = ROL2(Aki, 43);
		Amo ^= Do;
		BCo = ROL2(Amo, 21);
		Asu ^= Du;
		BCu = ROL2(Asu, 14);
		Eba = BCa ^ ((~BCe)&  BCi);
		Eba ^= vectorize(keccak_round_constants[round]);
		Ebe = BCe ^ ((~BCi)&  BCo);
		Ebi = BCi ^ ((~BCo)&  BCu);
		Ebo = BCo ^ ((~BCu)&  BCa);
		Ebu = BCu ^ ((~BCa)&  BCe);

		Abo ^= Do;
		BCa = ROL2(Abo, 28);
		Agu ^= Du;
		BCe = ROL2(Agu, 20);
		Aka ^= Da;
		BCi = ROL2(Aka, 3);
		Ame ^= De;
		BCo = ROL2(Ame, 45);
		Asi ^= Di;
		BCu = ROL2(Asi, 61);
		Ega = BCa ^ ((~BCe)&  BCi);
		Ege = BCe ^ ((~BCi)&  BCo);
		Egi = BCi ^ ((~BCo)&  BCu);
		Ego = BCo ^ ((~BCu)&  BCa);
		Egu = BCu ^ ((~BCa)&  BCe);

		Abe ^= De;
		BCa = ROL2(Abe, 1);
		Agi ^= Di;
		BCe = ROL2(Agi, 6);
		Ako ^= Do;
		BCi = ROL2(Ako, 25);
		Amu ^= Du;
		BCo = ROL2(Amu, 8);
		Asa ^= Da;
		BCu = ROL2(Asa, 18);
		Eka = BCa ^ ((~BCe)&  BCi);
		Eke = BCe ^ ((~BCi)&  BCo);
		Eki = BCi ^ ((~BCo)&  BCu);
		Eko = BCo ^ ((~BCu)&  BCa);
		Eku = BCu ^ ((~BCa)&  BCe);

		Abu ^= Du;
		BCa = ROL2(Abu, 27);
		Aga ^= Da;
		BCe = ROL2(Aga, 36);
		Ake ^= De;
		BCi = ROL2(Ake, 10);
		Ami ^= Di;
		BCo = ROL2(Ami, 15);
		Aso ^= Do;
		BCu = ROL2(Aso, 56);
		Ema = BCa ^ ((~BCe)&  BCi);
		Eme = BCe ^ ((~BCi)&  BCo);
		Emi = BCi ^ ((~BCo)&  BCu);
		Emo = BCo ^ ((~BCu)&  BCa);
		Emu = BCu ^ ((~BCa)&  BCe);

		Abi ^= Di;
		BCa = ROL2(Abi, 62);
		Ago ^= Do;
		BCe = ROL2(Ago, 55);
		Aku ^= Du;
		BCi = ROL2(Aku, 39);
		Ama ^= Da;
		BCo = ROL2(Ama, 41);
		Ase ^= De;
		BCu = ROL2(Ase, 2);
		Esa = BCa ^ ((~BCe)&  BCi);
		Ese = BCe ^ ((~BCi)&  BCo);
		Esi = BCi ^ ((~BCo)&  BCu);
		Eso = BCo ^ ((~BCu)&  BCa);
		Esu = BCu ^ ((~BCa)&  BCe);

		//    prepareTheta
		BCa = Eba^Ega^Eka^Ema^Esa;
		BCe = Ebe^Ege^Eke^Eme^Ese;
		BCi = Ebi^Egi^Eki^Emi^Esi;
		BCo = Ebo^Ego^Eko^Emo^Eso;
		BCu = Ebu^Egu^Eku^Emu^Esu;

		//thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
		Da = BCu^ROL2(BCe, 1);
		De = BCa^ROL2(BCi, 1);
		Di = BCe^ROL2(BCo, 1);
		Do = BCi^ROL2(BCu, 1);
		Du = BCo^ROL2(BCa, 1);

		Eba ^= Da;
		BCa = Eba;
		Ege ^= De;
		BCe = ROL2(Ege, 44);
		Eki ^= Di;
		BCi = ROL2(Eki, 43);
		Emo ^= Do;
		BCo = ROL2(Emo, 21);
		Esu ^= Du;
		BCu = ROL2(Esu, 14);
		Aba = BCa ^ ((~BCe)&  BCi);
		Aba ^= vectorize(keccak_round_constants[round + 1]);
		Abe = BCe ^ ((~BCi)&  BCo);
		Abi = BCi ^ ((~BCo)&  BCu);
		Abo = BCo ^ ((~BCu)&  BCa);
		Abu = BCu ^ ((~BCa)&  BCe);

		Ebo ^= Do;
		BCa = ROL2(Ebo, 28);
		Egu ^= Du;
		BCe = ROL2(Egu, 20);
		Eka ^= Da;
		BCi = ROL2(Eka, 3);
		Eme ^= De;
		BCo = ROL2(Eme, 45);
		Esi ^= Di;
		BCu = ROL2(Esi, 61);
		Aga = BCa ^ ((~BCe)&  BCi);
		Age = BCe ^ ((~BCi)&  BCo);
		Agi = BCi ^ ((~BCo)&  BCu);
		Ago = BCo ^ ((~BCu)&  BCa);
		Agu = BCu ^ ((~BCa)&  BCe);

		Ebe ^= De;
		BCa = ROL2(Ebe, 1);
		Egi ^= Di;
		BCe = ROL2(Egi, 6);
		Eko ^= Do;
		BCi = ROL2(Eko, 25);
		Emu ^= Du;
		BCo = ROL2(Emu, 8);
		Esa ^= Da;
		BCu = ROL2(Esa, 18);
		Aka = BCa ^ ((~BCe)&  BCi);
		Ake = BCe ^ ((~BCi)&  BCo);
		Aki = BCi ^ ((~BCo)&  BCu);
		Ako = BCo ^ ((~BCu)&  BCa);
		Aku = BCu ^ ((~BCa)&  BCe);

		Ebu ^= Du;
		BCa = ROL2(Ebu, 27);
		Ega ^= Da;
		BCe = ROL2(Ega, 36);
		Eke ^= De;
		BCi = ROL2(Eke, 10);
		Emi ^= Di;
		BCo = ROL2(Emi, 15);
		Eso ^= Do;
		BCu = ROL2(Eso, 56);
		Ama = BCa ^ ((~BCe)&  BCi);
		Ame = BCe ^ ((~BCi)&  BCo);
		Ami = BCi ^ ((~BCo)&  BCu);
		Amo = BCo ^ ((~BCu)&  BCa);
		Amu = BCu ^ ((~BCa)&  BCe);

		Ebi ^= Di;
		BCa = ROL2(Ebi, 62);
		Ego ^= Do;
		BCe = ROL2(Ego, 55);
		Eku ^= Du;
		BCi = ROL2(Eku, 39);
		Ema ^= Da;
		BCo = ROL2(Ema, 41);
		Ese ^= De;
		BCu = ROL2(Ese, 2);
		Asa = BCa ^ ((~BCe)&  BCi);
		Ase = BCe ^ ((~BCi)&  BCo);
		Asi = BCi ^ ((~BCo)&  BCu);
		Aso = BCo ^ ((~BCu)&  BCa);
		Asu = BCu ^ ((~BCa)&  BCe);
	}



	state[0] = devectorize(Aba);
	state[1] = devectorize(Abe);
	state[2] = devectorize(Abi);
	state[3] = devectorize(Abo);
	state[4] = devectorize(Abu);
	state[5] = devectorize(Aga);
	state[6] = devectorize(Age);
	state[7] = devectorize(Agi);
	state[8] = devectorize(Ago);
	state[9] = devectorize(Agu);
	state[10] = devectorize(Aka);
	state[11] = devectorize(Ake);
	state[12] = devectorize(Aki);
	state[13] = devectorize(Ako);
	state[14] = devectorize(Aku);
	state[15] = devectorize(Ama);
	state[16] = devectorize(Ame);
	state[17] = devectorize(Ami);
	state[18] = devectorize(Amo);
	state[19] = devectorize(Amu);
	state[20] = devectorize(Asa);
	state[21] = devectorize(Ase);
	state[22] = devectorize(Asi);
	state[23] = devectorize(Aso);
	state[24] = devectorize(Asu);


	//	if (thread == 0) {for (int i=0;i<25;i++) {printf("i%d uint2 %08x %08x\n",i, LOWORD(state[i]), HIWORD(state[i])); }}
}


__global__ __launch_bounds__(128,4) void  sk1024_keccak_gpu_hash(int threads, uint64_t startNonce, uint64_t *g_hash, uint64_t *g_nonceVector, uint64_t *resNounce)
{

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        
		uint64_t nonce = startNonce + thread;

         uint64_t state[25];
        
       #pragma unroll 9
	   for (int i = 0; i<9; i++) {state[i]=g_hash[i*threads+thread];}
	   #pragma unroll 16
	   for (int i = 9; i<25; i++) { state[i] = 0;}
       keccak_1600(state, RC);
       #pragma unroll 7
	   for (int i = 0; i<7; i++) { state[i] ^= g_hash[(9+i) * threads + thread]; }
	   state[7] ^= 0x05;
	   state[8] ^= (1ULL << 63);
	   keccak_1600(state, RC);
	   keccak_1600(state, RC);

	   if (state[6] <= pTarget[15]) { resNounce[0] = nonce; }

	} //thread
}


   
void sk1024_keccak_cpu_init(int thr_id, int threads)
{
    	
	cudaMemcpyToSymbol(RC,cpu_RC,sizeof(cpu_RC),0,cudaMemcpyHostToDevice);	
	cudaMalloc(&d_SKNonce[thr_id], sizeof(uint64_t));
	cudaMallocHost(&d_sknounce[thr_id], 1 * sizeof(uint64_t));
} 


__host__ uint64_t sk1024_keccak_cpu_hash(int thr_id, int threads, uint64_t startNounce, uint64_t *d_nonceVector, uint64_t *d_hash, int order)
{
	uint64_t result = 0xffffffffffffffff;
	cudaMemset(d_SKNonce[thr_id], 0xff, sizeof(uint64_t));
	int threadsperblock = 128;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

		sk1024_keccak_gpu_hash << <grid, block, shared_size >> >(threads, startNounce, d_hash, d_nonceVector, d_SKNonce[thr_id]);
	cudaMemcpy(d_sknounce[thr_id], d_SKNonce[thr_id], sizeof(uint64_t), cudaMemcpyDeviceToHost);
	
	result = *d_sknounce[thr_id];
	return result;
}
__host__ void sk1024_set_Target(const void *ptarget)
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol(pTarget, ptarget, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}
