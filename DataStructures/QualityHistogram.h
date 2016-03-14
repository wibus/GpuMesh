#ifndef GPUMESH_QUALITYHISTOGRAM
#define GPUMESH_QUALITYHISTOGRAM

#include <vector>


class QualityHistogram
{
public:
    QualityHistogram(std::size_t bucketCount = 100);
    virtual ~QualityHistogram();

    virtual void clear();

    std::size_t sampleCount() const;

    const std::vector<int>& buckets() const;
    void setBucket(std::size_t i, int count);

    double minimumQuality() const;
    void setMinimumQuality(double minimum);

    double averageQuality() const;
    void setAverageQuality(double average);

    std::size_t bucketCount() const;
    virtual void setBucketCount(std::size_t count);

    virtual void add(double value);

    virtual void merge(const QualityHistogram& histogram);

    virtual double computeGain(const QualityHistogram& reference) const;

private:
    std::size_t _sampleCount;
    double _minimumQuality;
    double _averageQuality;
    std::vector<int> _buckets;
};



// IMPLEMENTATION //
inline std::size_t QualityHistogram::sampleCount() const
{
    return _sampleCount;
}

inline const std::vector<int>& QualityHistogram::buckets() const
{
    return _buckets;
}

inline double QualityHistogram::minimumQuality() const
{
    return _minimumQuality;
}

inline double QualityHistogram::averageQuality() const
{
    return _averageQuality;
}

inline std::size_t QualityHistogram::bucketCount() const
{
    return _buckets.size();
}

#endif // GPUMESH_QUALITYHISTOGRAM
