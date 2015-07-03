#include "Timer.h"

namespace LLP
{
	Timer::Timer()
	{
		this->m_bStopped = false;
	}

	Timer::Timer(const Timer& timer)
	{
		this->m_bStopped = timer.GetIsStopped();
		this->TIMER_START = timer.GetTimerStart();
		this->TIMER_END = timer.GetTimerEnd();
	}

	Timer& Timer::operator=(const Timer& timer)
	{
		this->m_bStopped = timer.GetIsStopped();
		this->TIMER_START = timer.GetTimerStart();
		this->TIMER_END = timer.GetTimerEnd();

		return *this;
	}

	Timer::~Timer()
	{
	}

	void Timer::Start()
	{
		TIMER_START = boost::posix_time::microsec_clock::local_time();
		m_bStopped = false;
	}
		
	void Timer::Reset()
	{
		Start();
	}
		
	void Timer::Stop()
	{
		TIMER_END = boost::posix_time::microsec_clock::local_time();
		m_bStopped = true;
	}
		
	unsigned int Timer::Elapsed()
	{
		if(m_bStopped)
			return (TIMER_END - TIMER_START).total_seconds();
				
		return (boost::posix_time::microsec_clock::local_time() - TIMER_START).total_seconds();
	}
		
	unsigned int Timer::ElapsedMilliseconds()
	{
		if(m_bStopped)
			return (TIMER_END - TIMER_START).total_milliseconds();
				
		return (boost::posix_time::microsec_clock::local_time() - TIMER_START).total_milliseconds();
	}
}