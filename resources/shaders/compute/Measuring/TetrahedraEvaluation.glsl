layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform int TetCount;
uniform float MaxQuality;


struct Qual
{
    uint min;
    uint mean;
};

layout(shared, binding = 6) buffer Quals
{
    Qual quals[];
};


float tetQuality(Tet tet);


void main()
{
    uint uid = gl_GlobalInvocationID.x;
    uint gid = gl_WorkGroupID.x;


    if(uid < TetCount)
    {
        float q = tetQuality(tets[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(quals[gid].min, qi);
        atomicAdd(quals[gid].mean, qi);
    }
}
