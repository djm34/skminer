/*
* test routine for new algorithm
*
*/
#include "../hash/uint1024.h"
#include "hash/skein.h"
#include "hash/KeccakHash.h"

#include "miner2.h"

extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen
static uint64_t *d_hash[8];

extern void skein1024_cpu_init(int thr_id, int threads);
extern void skein1024_setBlock(void *pdata);
extern void skein1024_cpu_hash(int thr_id, int threads, uint64_t startNounce, uint64_t *d_hash, int order);
//    extern void sk1024_keccak_cpu_hash(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order)
extern uint64_t sk1024_keccak_cpu_hash(int thr_id, int threads, uint64_t startNounce, uint64_t *d_nonceVector, uint64_t *d_hash, int order);
extern void sk1024_keccak_cpu_init(int thr_id, int threads);
extern void sk1024_set_Target(const void *ptarget);

extern bool opt_benchmark;

bool scanhash_sk1024(unsigned int thr_id, unsigned int* TheData, uint1024 TheTarget, uint64_t &TheNonce, uint64_t max_nonce, uint64_t *hashes_done)
{
    uint64_t *ptarget = (uint64_t*)&TheTarget;
	
	const uint64_t first_nonce = TheNonce;

	const uint64_t Htarg = ptarget[15];

//	const int throughput = 2560 * 512 * 4;
	const int throughput = 512*8 * 512 * 4;
	//	if (first_nonce>max_nonce){max_nonce=pdata[19]+throughput;}
	static bool init[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 1 * 16 * sizeof(uint64_t) * throughput);
		skein1024_cpu_init(thr_id, throughput);
		sk1024_keccak_cpu_init(thr_id, throughput);
		init[thr_id] = true;
		
	}


	skein1024_setBlock((void*)TheData);
	sk1024_set_Target(ptarget);
//	do {

		int order = 0;
		skein1024_cpu_hash(thr_id, throughput, ((uint64_t*)TheData)[26], d_hash[thr_id], order++);
		uint64_t foundNonce = sk1024_keccak_cpu_hash(thr_id, throughput, ((uint64_t*)TheData)[26], NULL, d_hash[thr_id], order++);
		if (foundNonce != 0xffffffffffffffff)
		{


			((uint64_t*)TheData)[26] = foundNonce;
			/*
			uint1024 skein;
			Skein1024_Ctxt_t ctx;
			Skein1024_Init(&ctx, 1024);
			Skein1024_Update(&ctx, (unsigned char *)TheData, 216);
			Skein1024_Final(&ctx, (unsigned char *)&skein);

			uint64_t keccak[16];
			Keccak_HashInstance ctx_keccak;
			Keccak_HashInitialize(&ctx_keccak, 576, 1024, 1024, 0x05);
			Keccak_HashUpdate(&ctx_keccak, (unsigned char *)&skein, 1024);
			Keccak_HashFinal(&ctx_keccak, (unsigned char *)&keccak);
            */
//			if (keccak[15] <= Htarg) {
				TheNonce = foundNonce; //return the nonce
				*hashes_done = foundNonce - first_nonce + 1; 
				return true;
//			}
//			else {
//				printf("GPU #%d: result for nonce $%08X does not validate on CPU! \n", thr_id, foundNonce);
//			}
		}
		((uint64_t*)TheData)[26] += throughput;

//	} while (((uint64_t*)TheData)[26] < max_nonce);
	*hashes_done = ((uint64_t*)TheData)[26] - first_nonce + 1;
	return false;
}
