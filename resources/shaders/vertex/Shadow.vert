#version 440

uniform mat4 PVmat;
uniform vec4 CutPlaneEq;

layout(location=0) in vec3 position;

out float dist;


void main(void)
{
    gl_Position = PVmat * vec4(position, 1);
    dist = max(dot(position, CutPlaneEq.xyz) - CutPlaneEq.w, 0);
}
