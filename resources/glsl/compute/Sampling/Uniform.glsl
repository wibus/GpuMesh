uniform float MetricScaling;

mat3 metricAt(in vec3 position, inout uint cachedRefTet)
{
    return mat3(MetricScaling * MetricScaling);
}
