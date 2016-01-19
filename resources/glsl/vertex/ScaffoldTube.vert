#version 440

uniform mat4 ProjInv;
uniform mat4 ProjMat;
uniform mat4 ViewMat;
uniform vec2 Viewport;
uniform vec4 CutPlaneEq;
uniform float TubeRadius;


layout(location=0) in vec3 position;
layout(location=1) in float quality;


out vec3 wPos;
flat out vec3 wEnd;
flat out vec2 pEnd;
noperspective out vec2 pPos;
out float dept;
out float dist;
out float qual;


void main(void)
{
    wPos = position;
    wEnd = position;
    qual = quality;

    dist = dot(position, CutPlaneEq.xyz) - CutPlaneEq.w;

    vec4 pos4 = vec4(position, 1);
    vec4 eye4 = ViewMat * pos4;
    vec4 clip = ProjMat * eye4;
    gl_Position = clip;

    clip = clip/clip.w;
    float clipz = clip.z;

    vec3 view = (clip.xyz + 1.0) / 2.0;
    pPos = Viewport * view.xy;
    pEnd = pPos;

    float clipRadius = (TubeRadius / Viewport.x) * 2.0;
    vec4 eyeSize = ProjInv * vec4(clipRadius, 0.0, clipz, 1.0);
    float eyeRadius = eyeSize.x / eyeSize.w;

    vec4 clipSize = ProjMat * vec4(0.0, 0.0, eye4.z + eyeRadius, 1.0);
    float viewDepth = (clipSize.z / clipSize.w + 1.0) / 2.0;
    dept = viewDepth - view.z;
}
