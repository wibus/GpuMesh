layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(shared, binding = FIRST_FREE_BUFFER_BINDING) buffer Quals
{
    int qualMin;
    int means[];
};

float tetQuality(Tet tet);
float priQuality(Pri pri);
float hexQuality(Hex hex);

const int UINT_MAX = 2147483647;
const float MEAN_MAX = UINT_MAX / (gl_WorkGroupSize.x * 3);


void main()
{
    uint uid = gl_GlobalInvocationID.x;
    uint gid = gl_WorkGroupID.x;


    if(uid < pris.length())
    {
        float q = priQuality(pris[uid]);
        atomicMin(qualMin, int(q * UINT_MAX));
        atomicAdd(means[gid], int(q * MEAN_MAX));
    }
}
