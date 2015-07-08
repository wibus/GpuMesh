#version 440

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int HexCount;
uniform float MaxQuality;


struct Hex
{
    int v[8];
};

struct Qual
{
    uint min;
    uint mean;
};


layout(shared, binding = 3) buffer Hexs
{
    Hex hexs[];
};

layout(shared, binding = 4) buffer Quals
{
    Qual quals[];
};


float hexQuality(Hex hex);


void main()
{
    uint uid = gl_GlobalInvocationID.x;
    uint gid = gl_WorkGroupID.x;


    if(uid < HexCount)
    {
        float q = hexQuality(hexs[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(quals[gid].min, qi);
        atomicAdd(quals[gid].mean, qi);
    }
}
