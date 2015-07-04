#include "MinerThread.h"
#include "CBlock.h"
#include "../hash/templates.h"	
#include "bignum.h"
#include "../util_llh.h"
#include "../miner2.h"
namespace Core
{
	MinerThread::MinerThread(unsigned int pData)
	{
		m_pBLOCK = NULL;
		m_GpuId = pData;
		m_bBlockFound =false; 
		m_unHashes = 0;
		m_pTHREAD = boost::thread(&MinerThread::SK1024Miner, this);
		total_mhashes_done = 0;
	}

	MinerThread::MinerThread(const MinerThread& miner)
	{
		m_pBLOCK = miner.GetBlock();
		m_GpuId = miner.GetGpuId();
		m_bBlockFound = miner.GetIsBlockFound(); 
		m_bNewBlock = miner.GetIsNewBlock();
		m_bReady = miner.GetIsReady();
		m_unHashes = miner.GetHashes();
		m_pTHREAD = boost::thread(&MinerThread::SK1024Miner, this);
	}

	MinerThread& MinerThread::operator=(const MinerThread& miner)
	{
		m_pBLOCK = miner.GetBlock();
		m_GpuId = miner.GetGpuId();
		m_bBlockFound = miner.GetIsBlockFound(); 
		m_bNewBlock = miner.GetIsNewBlock();
		m_bReady = miner.GetIsReady();
		m_unHashes = miner.GetHashes();
		m_pTHREAD = boost::thread(&MinerThread::SK1024Miner, this);

		return *this;
	}

	MinerThread::~MinerThread()
	{
	}

 
	void MinerThread::SK1024Miner()
	{
		loop
		{
			try
			{
				/** Don't mine if the Connection Thread is Submitting Block or Receiving New. **/
				while(m_bNewBlock || m_bBlockFound || !m_pBLOCK)
					Sleep(1);
		
				CBigNum target;
				target.SetCompact(m_pBLOCK->GetBits());
	//			target.SetCompact(0x7e003fff); //simulate lower difficulty
				
				while(!m_bNewBlock)
				{					
					
					uint1024 hash;
					uint64_t hashes=0;
					unsigned int * TheData =(unsigned int*) m_pBLOCK->GetData();
					uint1024 TheTarget = target.getuint1024();
					uint64_t Nonce; //= m_pBLOCK->GetNonce();
					bool found = false;
					m_clLock.lock();
					{
						Nonce = m_pBLOCK->GetNonce();
				{	found = scanhash_sk1024(m_GpuId, TheData, TheTarget, Nonce, 512 * 8 * 512 * 40, &hashes);}
                        SetHashes(GetHashes()+hashes);
						
					}
					m_clLock.unlock();
					
					if(found)
					{
						m_bBlockFound = true;
						
						m_clLock.lock();
						{
							m_pBLOCK->SetNonce(Nonce);
						}
						m_clLock.unlock();
                        						
                        break;
					}
 //                   printf("hashes %d m_unHashes %d gethashes %d\n",hashes,m_unHashes,GetHashes());
					/*
					m_clLock.lock();
					{
						m_pBLOCK->SetNonce(m_pBLOCK->GetNonce()+hashes); 
					}
					m_clLock.unlock();

*/
					if(Nonce >= MAX_THREADS) //max_thread==> max_nonce
					{
						m_bNewBlock = true;
						break;
					}
				}
				
			}
			catch(std::exception& e)
			{ 
				printf("ERROR: %s\n", e.what()); 
			}
		}
	}


}