mat3 vertMetric(in vec3 position);

mat3 metricAt(in vec3 position, inout uint cachedRefTet)
{
    return vertMetric(position);
}
