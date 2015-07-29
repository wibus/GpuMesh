layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform float MaxQuality;

layout(shared, binding = FIRST_FREE_BUFFER_BINDING) buffer Quals
{
    uint qualMin;
    uint means[];
};


float tetQuality(Tet tet);
float priQuality(Pri pri);
float hexQuality(Hex hex);


void main()
{
    uint uid = gl_GlobalInvocationID.x;
    uint gid = gl_WorkGroupID.x;

    if(uid < tets.length())
    {
        float q = tetQuality(tets[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(qualMin, qi);
        atomicAdd(means[gid], qi);
    }

    if(uid < pris.length())
    {
        float q = priQuality(pris[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(qualMin, qi);
        atomicAdd(means[gid], qi);
    }

    if(uid < hexs.length())
    {
        float q = hexQuality(hexs[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(qualMin, qi);
        atomicAdd(means[gid], qi);
    }
}
