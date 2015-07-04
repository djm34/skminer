#ifndef __MINER2_H__
#define __MINER2_H__


#include "hash/uint1024.h"

/*
#ifdef __cplusplus
extern "C" {
#endif
*/
#include "cpuminer-config.h"

#include <stdbool.h>
#include <inttypes.h>

extern bool scanhash_sk1024(unsigned int thr_id, unsigned int* TheData, uint1024 TheTarget, uint64_t &TheNonce, uint64_t max_nonce, uint64_t  *hashes_done);
/*
#ifdef __cplusplus
}
#endif
*/
#endif /* __MINER2_H__ */
