#version 440

uniform mat4 PVmat;
uniform vec4 CutPlaneEq;

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 edge;
layout(location=3) in float quality;

out vec3 pos;
out vec3 nrm;
out vec3 edg;
out float qual;
out float dist;


void main(void)
{
    gl_Position = PVmat * vec4(position, 1);

    pos = position;
    nrm = normal;
    edg = edge;
    qual = quality;
    dist = max(dot(position, CutPlaneEq.xyz) - CutPlaneEq.w, 0);
}

