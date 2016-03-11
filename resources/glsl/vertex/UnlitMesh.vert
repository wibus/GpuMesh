#version 440

uniform mat4 ProjMat;
uniform mat4 ViewMat;
uniform vec4 CutPlaneEq;

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in float quality;

out vec3 eye;
out vec3 nrm;
out float qual;
out float dist;


void main(void)
{
    vec4 pos4 = vec4(position, 1);
    vec4 eye4 = ViewMat * pos4;
    gl_Position = ProjMat * eye4;

    eye = eye4.xyz;
    nrm = normal;
    qual = quality;
    dist = dot(position, CutPlaneEq.xyz) - CutPlaneEq.w;
}

