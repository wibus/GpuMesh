#version 440

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int PriCount;
uniform float MaxQuality;


struct Pri
{
    int v[6];
};

struct Qual
{
    uint min;
    uint mean;
};


layout(shared, binding = 2) buffer Pris
{
    Pri pris[];
};

layout(shared, binding = 4) buffer Quals
{
    Qual quals[];
};


float priQuality(Pri pri);


void main()
{
    uint uid = gl_GlobalInvocationID.x;
    uint gid = gl_WorkGroupID.x;


    if(uid < PriCount)
    {
        float q = priQuality(pris[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(quals[gid].min, qi);
        atomicAdd(quals[gid].mean, qi);
    }
}
