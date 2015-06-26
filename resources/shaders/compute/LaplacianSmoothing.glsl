#version 440

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int VertCount;
uniform float MoveCoeff;

layout (std140, binding=0) buffer Vert
{
    vec4 p[];
} vert;

struct TopoStruct
{
    int type;
    int base;
    int count;
    int pad;
};

layout (std140, binding=1) buffer Topo
{
    TopoStruct t[];
} topo;

layout (std140, binding=2) buffer Neig
{
    int n[];
} neig;


vec3 snaptoBoundary(int boundaryID, vec3 pos);

void main()
{
    uvec3 wholeGridSize = gl_NumWorkGroups * gl_WorkGroupSize;
    uint uid = gl_GlobalInvocationID.z * wholeGridSize.x * wholeGridSize.y +
               gl_GlobalInvocationID.z * wholeGridSize.x +
               gl_GlobalInvocationID.x;


    // Read
    vec3 pos = vec3(vert.p[uid]);


    // Modification
    int type = topo.t[uid].type;
    if(type >= 0)
    {
        float weightSum = 0.0;
        vec3 barycenter = vec3(0.0);

        int n = topo.t[uid].base;
        int count = topo.t[uid].count;

        for(int i=0; i<count; ++i, ++n)
        {
            vec3 npos = vec3(vert.p[neig.n[n]]);
            float weight = length(npos - pos) + 0.0001;
            float alpha = weight / (weightSum + weight);
            barycenter = mix(barycenter, npos, alpha);
            weightSum += weight;
        }

        pos = mix(pos, barycenter, MoveCoeff);

        if(type > 0)
        {
            pos = snaptoBoundary(type, pos);
        }
    }


    // Write
    vert.p[uid] = vec4(pos, 0.0);
}
