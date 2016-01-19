#version 450

layout (local_size_x = 1, local_size_y = 256, local_size_z = 1) in;


uniform float MinHeight; // [0.0, 1.0]

layout(shared, binding = 0) buffer Color
{
    vec4 colors[];
};


vec3 qualityLut(in float q);


void main()
{
    float height = float(gl_LocalInvocationID.y) / float(gl_WorkGroupSize.y-1);
    const uint columnSize = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    colors[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * columnSize] =
        vec4(qualityLut((height - MinHeight) / (1.0 - MinHeight)), 1.0);
}
