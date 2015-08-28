#include "VertexAccum.h"


IVertexAccum::IVertexAccum()
{

}

IVertexAccum::~IVertexAccum()
{

}



NotThreadSafeVertexAccum::NotThreadSafeVertexAccum() :
    _posAccum(),
    _weightAccum(0.0)
{

}

NotThreadSafeVertexAccum::~NotThreadSafeVertexAccum()
{

}

void NotThreadSafeVertexAccum::add(const glm::dvec3 pos, double weight)
{
    _posAccum += pos * weight;
    _weightAccum += weight;
}

bool NotThreadSafeVertexAccum::assignAverage(glm::dvec3& pos) const
{
    if(_weightAccum != 0.0)
    {
        pos = _posAccum / _weightAccum;
        return true;
    }
    return false;
}


ThreadSafeVertexAccum::ThreadSafeVertexAccum() :
    _posAccum(),
    _weightAccum(0.0)
{

}

ThreadSafeVertexAccum::~ThreadSafeVertexAccum()
{

}

void ThreadSafeVertexAccum::add(const glm::dvec3 pos, double weight)
{
    _mutex.lock();
    _posAccum += pos * weight;
    _weightAccum += weight;
    _mutex.unlock();
}

bool ThreadSafeVertexAccum::assignAverage(glm::dvec3& pos) const
{
    _mutex.lock();
    if(_weightAccum != 0.0)
    {
        pos = _posAccum / _weightAccum;
        _mutex.unlock();
        return true;
    }
    _mutex.unlock();
    return false;
}
