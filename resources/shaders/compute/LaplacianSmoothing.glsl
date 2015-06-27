#version 440

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int VertCount;
uniform float MoveCoeff;


struct TopoStruct
{
    int type;
    int base;
    int count;

    // Dummy int for 4N alignment (std140 layout)
    int padding;
};


// Vertices positions
// vec4 is only for alignment
layout (std140, binding=0) buffer Vert
{
    vec4 vert[];
};

// Topology indirection table
// type == -1 : fixed vertex
// type ==  0 : free vertex
// type  >  0 : boundary vertex
// When type > 0, type is boundary's ID
layout (std140, binding=1) buffer Topo
{
    TopoStruct topo[];
};

// Adjacency lists buffer
// (!) packed layout makes int array tightly packed
// It may not be the case for every machine (!)
layout (packed, binding=2) buffer Neig
{
    int neig[];
};


vec3 snaptoBoundary(int boundaryID, vec3 pos);

void main()
{
    uvec3 wholeGridSize = gl_NumWorkGroups * gl_WorkGroupSize;
    uint uid = gl_GlobalInvocationID.z * wholeGridSize.x * wholeGridSize.y +
               gl_GlobalInvocationID.z * wholeGridSize.x +
               gl_GlobalInvocationID.x;


    // Read
    vec3 pos = vec3(vert[uid]);


    // Modification
    int type = topo[uid].type;
    int count = topo[uid].count;
    if(type >= 0 && count > 0)
    {
        float weightSum = 0.0;
        vec3 barycenter = vec3(0.0);

        int n = topo[uid].base;
        for(int i=0; i<count; ++i, ++n)
        {
            vec3 npos = vec3(vert[neig[n]]);

            vec3 dist = npos - pos;
            float weight = dot(dist, dist) + 0.0001;

            barycenter += npos * weight;
            weightSum += weight;
        }

        barycenter /= weightSum;
        pos = mix(pos, barycenter, MoveCoeff);

        if(type > 0)
        {
            pos = snaptoBoundary(type, pos);
        }
    }


    // Write
    vert[uid] = vec4(pos, 0.0);
}
