#version 440

layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

uniform int VertCount;
uniform float MoveCoeff;

layout (std140, binding=0) buffer Vert
{
    vec4 p[];
} vert;

layout (std140, binding=0) buffer Adja
{
    vec4 n[];
} adja;


void main()
{
    uvec3 wholeGridSize = gl_NumWorkGroups * gl_WorkGroupSize;
    uint uid = gl_GlobalInvocationID.z * wholeGridSize.x * wholeGridSize.y +
               gl_GlobalInvocationID.z * wholeGridSize.x +
               gl_GlobalInvocationID.x;

    vec4 pos;
    if(uid < VertCount)
    {
        // Read
        pos = vert.p[uid];


        // Modification
        pos = pos * MoveCoeff;
    }


    // Synchronize
    barrier();


    if(uid < VertCount)
    {
        // Write
        vert.p[uid] = pos;
    }
}
