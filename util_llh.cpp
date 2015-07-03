#include "util_llh.h"

/** Find out how many cores the CPU has. **/
int GetTotalCores()
{
	int nProcessors = boost::thread::hardware_concurrency();
	if (nProcessors < 1)
		nProcessors = 1;
			
	return nProcessors;
}

