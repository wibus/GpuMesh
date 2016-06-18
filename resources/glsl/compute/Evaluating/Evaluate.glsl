layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(shared, binding = EVALUATE_QUAL_BUFFER_BINDING) buffer Quals
{
    int qualMin;
    float invLogSum;
};

layout(shared, binding = EVALUATE_HIST_BUFFER_BINDING) buffer Hists
{
    int hists[];
};

uniform float hists_length;

const int MIN_MAX = 2147483647;

float tetQuality(inout Tet tet);
float priQuality(inout Pri pri);
float hexQuality(inout Hex hex);



void commit(uint gid, float q)
{
    atomicMin(qualMin, int(q * MIN_MAX));
    atomicAdd(invLogSum, log(1.0 / q));

    // ! Driver linker bug :
    // Makes the linker segfautl
    //  float histCount = float(hists.length())
    int bucket = int(max(q * hists_length, 0.0));
    atomicAdd(hists[bucket], 1);
}

void main()
{
    uint vId = gl_GlobalInvocationID.x;
    uint gid = gl_WorkGroupID.x;


    if(vId < tets.length())
    {
        commit( gid, tetQuality(tets[vId]) );
    }

    if(vId < pris.length())
    {
        commit( gid, priQuality(pris[vId]) );
    }

    if(vId < hexs.length())
    {
        commit( gid, hexQuality(hexs[vId]) );
    }
}
