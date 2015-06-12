#version 440

uniform mat4 PVmat;
uniform mat4 PVshadow;
uniform vec4 CutPlaneEq;

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 edge;
layout(location=3) in float quality;

out vec3 pos;
out vec3 lgt;
out vec3 nrm;
out vec3 edg;
out float qual;
out float dist;


void main(void)
{
    vec4 position4 = vec4(position, 1);
    gl_Position = PVmat * position4;
    vec4 lgt4 = PVshadow * position4;
    lgt = lgt4.xyz / lgt4.w;

    pos = position;
    nrm = normal;
    edg = edge;
    qual = quality;
    dist = dot(position, CutPlaneEq.xyz) - CutPlaneEq.w;
}

