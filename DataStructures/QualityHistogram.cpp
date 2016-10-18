#include "QualityHistogram.h"

#include <GLM/glm.hpp>


QualityHistogram::QualityHistogram() :
    _buckets(40, 0),
    _sampleCount(0),
    _minimumQuality(1.0),
    _invQualitySum(0.0)
{

}

QualityHistogram::QualityHistogram(std::size_t bucketCount) :
    _buckets(bucketCount, 0),
    _sampleCount(0),
    _minimumQuality(1.0),
    _invQualitySum(0.0)
{

}

QualityHistogram::~QualityHistogram()
{

}

void QualityHistogram::clear()
{
    _sampleCount = 0;
    _minimumQuality = 1.0;
    _invQualitySum = 0.0;
    std::fill(_buckets.begin(), _buckets.end(), 0);
}

void QualityHistogram::setSampleCount(std::size_t count)
{
    _sampleCount = count;
}

void QualityHistogram::setBucketCount(std::size_t count)
{
    _buckets.resize(count);
    clear();
}

void QualityHistogram::setBucket(std::size_t i, int count)
{
    _buckets[i] = count;
}

void QualityHistogram::setMinimumQuality(double minimum)
{
    _minimumQuality = minimum;
}

void QualityHistogram::setInvQualitySum(double sum)
{
    _invQualitySum = sum;
}

void QualityHistogram::add(double value)
{
    ++_sampleCount;
    _minimumQuality = glm::min(_minimumQuality, value);

    if(value > 0.0)
        _invQualitySum += 1.0 / value;
    else
        _invQualitySum = INFINITY;

    size_t bucketCount = _buckets.size();
    size_t b = glm::clamp(size_t(value * bucketCount), size_t(0), bucketCount-1);
    _buckets[b] += 1;
}

void QualityHistogram::merge(const QualityHistogram& histogram)
{
    assert(_buckets.size() == histogram._buckets.size());

    if(histogram.sampleCount() == 0)
        return;

    _sampleCount += histogram._sampleCount;
    _minimumQuality = glm::min(_minimumQuality, histogram._minimumQuality);
    _invQualitySum += histogram._invQualitySum;

    size_t bucketCount = _buckets.size();
    for(size_t i=0; i < bucketCount; ++i)
        _buckets[i] += histogram._buckets[i];
}

double QualityHistogram::computeGain(const QualityHistogram& reference) const
{
    return harmonicMean() - reference.harmonicMean();
}
