#version 440

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int TetCount;
uniform int PriCount;
uniform int HexCount;
uniform float MaxQuality;


struct Tet
{
    int v[4];
};

struct Pri
{
    int v[6];
};

struct Hex
{
    int v[8];
};

struct Qual
{
    uint min;
    uint mean;
};


layout(shared, binding = 1) buffer Tets
{
    Tet tets[];
};

layout(shared, binding = 2) buffer Pris
{
    Pri pris[];
};

layout(shared, binding = 3) buffer Hexs
{
    Hex hexs[];
};

layout(shared, binding = 4) buffer Quals
{
    Qual quals[];
};


float tetQuality(Tet tet);
float priQuality(Pri pri);
float hexQuality(Hex hex);


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

    if(uid < PriCount)
    {
        float q = priQuality(pris[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(quals[gid].min, qi);
        atomicAdd(quals[gid].mean, qi);
    }

    if(uid < HexCount)
    {
        float q = hexQuality(hexs[uid]);
        uint qi = uint(q * MaxQuality);
        atomicMin(quals[gid].min, qi);
        atomicAdd(quals[gid].mean, qi);
    }
}
