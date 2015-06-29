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


layout(shared, binding = 1) buffer Qual
{
    uint mean;
    uint var;
    uint min;
};

layout(shared, binding = 2) buffer Tets
{
    Tet tets[];
};

layout(shared, binding = 3) buffer Pris
{
    Pri pris[];
};

layout(shared, binding = 4) buffer Hexs
{
    Hex hexs[];
};


float tetQuality(Tet tet);
float priQuality(Pri pri);
float hexQuality(Hex hex);


void main()
{
    uint uid = gl_GlobalInvocationID.x;

    if(uid < TetCount)
    {
        float q = tetQuality(tets[uid]);
        uint qi = uint(q * MaxQuality);
        atomicAdd(mean, qi);
        atomicMin(min, qi);
    }

    if(uid < PriCount)
    {
        float q = priQuality(pris[uid]);
        uint qi = uint(q * MaxQuality);
        atomicAdd(mean, qi);
        atomicMin(min, qi);
    }

    if(uid < HexCount)
    {
        float q = hexQuality(hexs[uid]);
        uint qi = uint(q * MaxQuality);
        atomicAdd(mean, qi);
        atomicMin(min, qi);
    }
}
