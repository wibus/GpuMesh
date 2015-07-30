layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform uint GroupCount;


layout(shared, binding = FIRST_FREE_BUFFER_BINDING) buffer Quals
{
    uint qualMin;
    uint means[];
};

layout(shared, binding = FIRST_FREE_BUFFER_BINDING + 1) buffer Mean
{
    uint mean;
};


void main()
{
    // Looks like this algorithm needs more than 1M elements
    // to beat CPU serial summation.

    uint uid = gl_GlobalInvocationID.x;

    uint groupBeg = GroupCount * uid;
    uint groupEnd = min(groupBeg + GroupCount, means.length());

    float meanSum = 0.0;
    for(uint i = groupBeg; i < groupEnd; ++i)
    {
        meanSum += means[i];
    }
    meanSum /= gl_WorkGroupSize.x;

    atomicAdd(mean, uint(meanSum));
}
