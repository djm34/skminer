/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2014 pooler
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "cpuminer-config.h"
#define _GNU_SOURCE

#include "core.h"
#include "hash/templates.h"
#include "hash/CBlock.h"
#include "hash/Miner.h"
#include "hash/MinerThread.h"

unsigned int nBlocksFoundCounter = 0;
unsigned int nBlocksAccepted = 0;
unsigned int nBlocksRejected = 0;
unsigned int nDifficulty = 0;
unsigned int nBestHeight = 0;
unsigned int nStartTimer = 0;
unsigned int nBaseDiffBits = 1007; //1007-bit
unsigned int nBaseDiff = 2122317823;
unsigned int nBitCount = 0;

bool isBlockSubmission = false;
bool isNewDifficulty = false;
double  nHashes = 0.0;

CBigNum bnTarget = 0;

char *device_name[8]; // CB
namespace Core
{
	/** Class to handle all the Connections via Mining LLP.
	Independent of Mining Threads for Higher Efficiency. **/
	class ServerConnection
	{
	public:
		LLP::Miner* CLIENT;
		int nThreads, nTimeout;
		std::vector<MinerThread*> THREADS;
		LLP::Thread_t THREAD;
		LLP::Timer    TIMER;
		std::string   IP, PORT;

		ServerConnection(std::string ip, std::string port, int nMaxThreads, int nMaxTimeout) : IP(ip), PORT(port), TIMER(), nThreads(nMaxThreads), nTimeout(nMaxTimeout), THREAD(boost::bind(&ServerConnection::ServerThread, this))
		{
			for (int nIndex = 0; nIndex < nThreads; nIndex++)
				THREADS.push_back(new MinerThread(nIndex));
		}

		/** Reset the block on each of the Threads. **/
		void ResetThreads()
		{

			/** Reset each individual flag to tell threads to stop mining. **/
			for (unsigned int nIndex = 0; nIndex < THREADS.size(); nIndex++)
			{
				THREADS[nIndex]->SetIsBlockFound(false);
				THREADS[nIndex]->SetIsNewBlock(true);
			}

		}

		unsigned long long Hashes()
		{
			unsigned long long nHashes = 0;
			for (unsigned int nIndex = 0; nIndex < THREADS.size(); nIndex++)
			{
				nHashes += THREADS[nIndex]->GetHashes();
			}

			return nHashes;
		}

		/** Main Connection Thread. Handles all the networking to allow
		Mining threads the most performance. **/
		void ServerThread()
		{

			/** Don't begin until all mining threads are Created. **/
			while (THREADS.size() != nThreads)
				Sleep(1);


			/** Initialize the Server Connection. **/
			CLIENT = new LLP::Miner(IP, PORT);


			/** Initialize a Timer for the Hash Meter. **/
			TIMER.Start();

			//unsigned int nBestHeight = 0;
			loop
			{
				try
				{
					/** Run this thread at 1 Cycle per Second. **/
					Sleep(1000);


					/** Attempt with best efforts to keep the Connection Alive. **/
					if (!CLIENT->Connected() || CLIENT->Errors())
					{
						ResetThreads();

						if (!CLIENT->Connect())
							continue;
						else
							CLIENT->SetChannel(2);
					}

					/** Check the Block Height. **/
					unsigned int nHeight = CLIENT->GetHeight(nTimeout);
					if (nHeight == 0)
					{
						printf("Failed to Update Height...\n");
						CLIENT->Disconnect();
						continue;
					}

					/** If there is a new block, Flag the Threads to Stop Mining. **/
					if (nHeight != nBestHeight)
					{
						isBlockSubmission = false;
						nBestHeight = nHeight;
						printf("[MASTER] Coinshield Network: New Block %u\n", nHeight);

						isNewDifficulty = true;
						ResetThreads();
					}


					/** Rudimentary Meter **/
					if (TIMER.Elapsed() > 10)
					{
						unsigned int nSecondsElapsed = (unsigned int)time(0) - nStartTimer;

						nHashes = (double)Hashes() / 1000000.0;
						double MegaHashesPerSecond = nHashes / (double) nSecondsElapsed;
						unsigned int nDiff = 0;


						if (isNewDifficulty)
						{
							bnTarget.SetCompact(nDifficulty);

							nBitCount = 0;
							while (bnTarget > 0)
							{
								nBitCount++;
								bnTarget >>= 1;
							}
						}

						nDiff = (1024 - nBitCount);

						printf("%.1f MH/s | %u Blks ACC=%u REJ=%u | Height = %u | Diff = %u 0-bits | %02d:%02d:%02d\n",  MegaHashesPerSecond, nBlocksFoundCounter, nBlocksAccepted, nBlocksRejected, nBestHeight, nDiff, (nSecondsElapsed / 3600) % 60, (nSecondsElapsed / 60) % 60, (nSecondsElapsed) % 60);

						isNewDifficulty = false;
						TIMER.Reset();
					}


					/** Check if there is work to do for each Miner Thread. **/
					for (int nIndex = 0; nIndex < THREADS.size(); nIndex++)
					{
						if (THREADS[nIndex]->GetIsBlockFound())
						{
							if (THREADS[nIndex]->GetBlock()->GetHeight() == nBestHeight)
							{
								if (!isBlockSubmission)
								{
									isBlockSubmission = true;
									unsigned int nDiff = 0;

									bnTarget.SetCompact(nDifficulty);

									nBitCount = 0;
									while (bnTarget > 0)
									{
										nBitCount++;
										bnTarget >>= 1;
									}

									nDiff = (1024 - nBitCount);

									printf("\n\t\t%s (%u)\n", device_name[nIndex], nIndex + 1);
									printf("\n******* BLOCK with %u Leading Zero Bits FOUND *******\n", nDiff);
									nBlocksFoundCounter++;
								}
							}

							printf("\n\t\tPreparing for Block Submission...\n");
						}

						/** Attempt to get a new block from the Server if Thread needs One. **/
						if (THREADS[nIndex]->GetIsNewBlock())
						{
							/** Retrieve new block from Server. **/
							CBlock* pBlock = CLIENT->GetBlock(nTimeout);


							/** If the block is good, tell the Mining Thread its okay to Mine. **/
							if (pBlock)
							{
								THREADS[nIndex]->SetIsBlockFound(false);
								THREADS[nIndex]->SetIsNewBlock(false);
								THREADS[nIndex]->SetBlock(pBlock);

								nDifficulty = THREADS[nIndex]->GetBlock()->GetBits();
							}

							/** If the Block didn't come in properly, Reconnect to the Server. **/
							else
							{
								CLIENT->Disconnect();

								break;
							}

						}

						/** Submit a block from Mining Thread if Flagged. **/
						else if (THREADS[nIndex]->GetIsBlockFound())
						{
							printf("\nSubmitting Block...\n");

							if (!THREADS[nIndex]->GetBlock())
							{
								THREADS[nIndex]->SetIsNewBlock(true);
								THREADS[nIndex]->SetIsBlockFound(false);

								continue;
							}

							/** Attempt to Submit the Block to Network. **/
							unsigned char RESPONSE = CLIENT->SubmitBlock(THREADS[nIndex]->GetBlock()->GetMerkleRoot(), THREADS[nIndex]->GetBlock()->GetNonce(), nTimeout);

							/** Check the Response from the Server.**/
							if (RESPONSE == 200)
							{
								printf("[MASTER] Block Accepted By Coinshield Network.\n");

								ResetThreads();
								nBlocksAccepted++;
							}
							else if (RESPONSE == 201)
							{
								printf("[MASTER] Block Rejected by Coinshield Network.\n");

								THREADS[nIndex]->SetIsNewBlock(true);
								THREADS[nIndex]->SetIsBlockFound(false);

								isBlockSubmission = false;
								nBlocksRejected++;
							}

							/** If the Response was Bad, Reconnect to Server. **/
							else
							{
								printf("[MASTER] Failure to Submit Block. Reconnecting...\n");
								CLIENT->Disconnect();

								break;
							}
						}
					}
				}
				catch (std::exception& e)
				{
					printf("%s\n", e.what()); CLIENT = new LLP::Miner(IP, PORT);
				}
			}
		}
	};
}



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#ifdef WIN32

#include <windows.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/resource.h>
#if HAVE_SYS_SYSCTL_H
#include <sys/types.h>
#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <sys/sysctl.h>
#endif
#endif

#include "miner2.h"

#ifdef WIN32
#include <Mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

#define PROGRAM_NAME		"skminer"

// from heavy.cu
#ifdef __cplusplus
extern "C"
{
#endif
int cuda_num_devices();
void cuda_devicenames();
int cuda_finddevice(char *name);
void cuda_deviceproperties(int GPU_N);

#ifdef __cplusplus
}
#endif



int device_map[8] = { 0, 1, 2, 3, 4, 5, 6, 7 }; // CB
static int num_processors;
int tp_coef[8];











#ifndef WIN32
static void signal_handler(int sig)
{
	switch (sig) {
	case SIGHUP:
		applog(LOG_INFO, "SIGHUP received");
		break;
	case SIGINT:
		applog(LOG_INFO, "SIGINT received, exiting");
		exit(0);
		break;
	case SIGTERM:
		applog(LOG_INFO, "SIGTERM received, exiting");
		exit(0);
		break;
	}
}
#endif

#define PROGRAM_VERSION "v0.3"
int main(int argc, char *argv[])
{
	struct thr_info *thr;
	long flags;
//	int i;

#ifdef WIN32
	SYSTEM_INFO sysinfo;
#endif

	 printf("        ***** skMiner for nVidia GPUs by djm34  *****\n");
	 printf("\t             This is version "PROGRAM_VERSION" \n");
	 printf("	based on ccMiner by Christian Buchner and Christian H. 2014 ***\n");
	 printf("                   and on primeminer by Videlicet\n");
	 printf("\t Copyright 2014 djm34\n");
	 printf("\t  BTC donation address: 1NENYmxwZGHsKFmyjTc5WferTn5VTFb7Ze\n");
     printf("\t                   CSD: 2S2pCpRXyb8Lpre52U3Xjq2MguSdaea5YGjVTsJqgZBfL2S24ag\n");
	

	num_processors = cuda_num_devices();
	std::string TheURL = argv[1];
	std::string device;
	if (argc>2) {device = argv[2];}
	int posport = TheURL.rfind(":");
	std::string TheIP = TheURL.substr(0, posport);
	std::string ThePORT = TheURL.substr(posport + 1, TheURL.size());
    std::vector<std::string> Devices;
    std::vector<std::size_t> Position; 

	if (device.size()!=0){
num_processors=0;
		size_t pos = device.find(",");
        size_t lastpos =0;
	 while (pos!=std::string::npos)
{
		Devices.push_back(device.substr(lastpos,pos-lastpos));
        lastpos=pos+1;
		pos = device.find(",",lastpos);
};
	Devices.push_back(device.substr(lastpos, device.size()));

	printf("cards asked %d\n", Devices.size());

		for (int i = 0; i<Devices.size(); i++){
			int CardNum = cuda_finddevice((char*)Devices[i].c_str());
if (CardNum!=-1)	{device_map[num_processors] = CardNum;
                     num_processors++;
			printf("existing card: %s\n", Devices[i].c_str());}}
if (num_processors == 0) { printf("you wish you had those cards... \n"); exit(0);}else {cuda_deviceproperties(num_processors);} 

	} else {cuda_devicenames();}
if (num_processors == 0) { printf("No nvidia card found on the system... \n"); exit(0);  }


	printf("Initializing Miner %s:%s Number of Cards = %i Timeout = %i\n", TheIP.c_str(), ThePORT.c_str(), num_processors, 10);
	nStartTimer = (unsigned int)time(0);
	Core::ServerConnection MINERS(TheIP, ThePORT,num_processors, 10);
	loop{ Sleep(10); }


}
