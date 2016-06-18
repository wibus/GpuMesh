#ifndef GPUMESH_QUALITYHISTOGRAM
#define GPUMESH_QUALITYHISTOGRAM

#include <cmath>
#include <vector>


class QualityHistogram
{
public:
    QualityHistogram();
    QualityHistogram(std::size_t bucketCount);
    virtual ~QualityHistogram();


    virtual void clear();


    std::size_t sampleCount() const;
    void setSampleCount(std::size_t count);

    std::size_t bucketCount() const;
    virtual void setBucketCount(std::size_t count);

    const std::vector<int>& buckets() const;
    void setBucket(std::size_t i, int count);

    double minimumQuality() const;
    void setMinimumQuality(double minimum);

    double geometricMean() const;
    void setInvQualityLogSum(double sum);


    virtual void add(double value);

    virtual void merge(const QualityHistogram& histogram);

    virtual double computeGain(const QualityHistogram& reference) const;

private:
    std::size_t _sampleCount;
    double _minimumQuality;
    double _invQualityLogSum;
    std::vector<int> _buckets;
};



// IMPLEMENTATION //
inline std::size_t QualityHistogram::sampleCount() const
{
    return _sampleCount;
}

inline std::size_t QualityHistogram::bucketCount() const
{
    return _buckets.size();
}

inline const std::vector<int>& QualityHistogram::buckets() const
{
    return _buckets;
}

inline double QualityHistogram::minimumQuality() const
{
    return _minimumQuality;
}

inline double QualityHistogram::geometricMean() const
{
    return exp( - _invQualityLogSum / sampleCount());
}

#endif // GPUMESH_QUALITYHISTOGRAM
