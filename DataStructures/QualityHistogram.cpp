#include "QualityHistogram.h"

#include <GLM/glm.hpp>


QualityHistogram::QualityHistogram(std::size_t bucketCount) :
    _sampleCount(0),
    _minimumQuality(1.0),
    _averageQuality(0.0),
    _buckets(bucketCount, 0)
{

}

QualityHistogram::~QualityHistogram()
{

}

void QualityHistogram::clear()
{
    _sampleCount = 0;
    _minimumQuality = 1.0;
    _averageQuality = 0.0;
    std::fill(_buckets.begin(), _buckets.end(), 0);
}

void QualityHistogram::setBucket(std::size_t i, int count)
{
    _buckets[i] = count;
}

void QualityHistogram::setMinimumQuality(double minimum)
{
    _minimumQuality = minimum;
}

void QualityHistogram::setAverageQuality(double average)
{
    _averageQuality = average;
}

void QualityHistogram::setBucketCount(std::size_t count)
{
    _buckets.resize(count);
    clear();
}

void QualityHistogram::add(double value)
{
    ++_sampleCount;
    _minimumQuality = glm::min(_minimumQuality, value);
    _averageQuality = glm::mix(_averageQuality, value, 1.0 / _sampleCount);

    size_t bucketCount = _buckets.size();
    size_t b = glm::clamp(size_t(value * bucketCount), size_t(0), bucketCount-1);
    _buckets[b] += 1;
}

void QualityHistogram::merge(const QualityHistogram& histogram)
{
    assert(_buckets.size() == histogram._buckets.size());

    _sampleCount += histogram._sampleCount;
    _minimumQuality = glm::min(_minimumQuality, histogram._minimumQuality);
    _averageQuality = glm::mix(_averageQuality, histogram._averageQuality,
                               double(histogram._sampleCount) / _sampleCount);

    size_t bucketCount = _buckets.size();
    for(size_t i=0; i < bucketCount; ++i)
        _buckets[i] += histogram._buckets[i];
}

double QualityHistogram::computeGain(const QualityHistogram& reference) const
{
    assert(_buckets.size() == reference._buckets.size());

    size_t bucketCount = _buckets.size();
    std::vector<int> diff(bucketCount);
    for(size_t i=0; i < bucketCount; ++i)
        diff[i] = _buckets[i] - reference._buckets[i];

    double gain = 0.0;
    for(size_t i=0; i < bucketCount; ++i)
    {
        double d = diff[i];
        double w = double(bucketCount) - i;
        gain += d*w;
    }

    return gain / _sampleCount;
}
