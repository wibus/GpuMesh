mat3 vertMetric(in vec3 position);

mat3 metricAt(in vec3 position, in uint vId)
{
    return vertMetric(position);
}
