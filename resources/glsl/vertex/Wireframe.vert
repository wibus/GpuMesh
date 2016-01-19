#version 440

uniform mat4 ProjViewMat;
uniform vec4 CutPlaneEq;


layout(location = 0) in vec3 position;
layout(location = 1) in float quality;


out float qual;
out float dist;


void main()
{
    dist = dot(position, CutPlaneEq.xyz) - CutPlaneEq.w;
    gl_Position = ProjViewMat * vec4(position, 1.0);
    qual = quality;
}
