struct VertexAccum
{
    vec3 posAccum;
    float weightAccum;
};

layout(shared, binding = VERTEX_ACCUMS_BUFFER_BINDING) buffer VertexAccums
{
    VertexAccum vertexAccums[];
};


void addPosition(in uint vId, in vec3 pos, in float weight)
{
    vec3 val = pos * weight;
    atomicAdd(vertexAccums[vId].posAccum.x, val.x);
    atomicAdd(vertexAccums[vId].posAccum.y, val.y);
    atomicAdd(vertexAccums[vId].posAccum.z, val.z);
    atomicAdd(vertexAccums[vId].weightAccum, weight);
}

bool assignAverage(in uint vId, inout vec3 pos)
{
    float weightAccum = vertexAccums[vId].weightAccum;
    if(weightAccum > 0.0)
    {
        pos = vertexAccums[vId].posAccum / weightAccum;
        return true;
    }
    return false;
}

void reinitAccum(in uint vId)
{
    vertexAccums[vId].posAccum = vec3(0.0);
    vertexAccums[vId].weightAccum  = 0.0;
}
