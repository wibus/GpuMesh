#version 440

uniform mat4 ProjInv;
uniform mat4 ProjMat;
uniform mat4 ViewMat;
uniform vec2 Viewport;
uniform vec4 CutPlaneEq;
uniform float JointTubeMinRatio;
uniform float JointRadius;
uniform float TubeRadius;

layout(location=0) in vec3 position;
layout(location=1) in float quality;

out vec3 wPos;
noperspective out vec2 pPos;
out float qual;
out float size;
out float dept;
out float dist;


void main(void)
{
    wPos = position;
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


    float clipMinRadius = (TubeRadius * JointTubeMinRatio / Viewport.x) * 2.0;
    vec4 eyeMinSize = ProjInv * vec4(clipMinRadius, 0.0, clipz, 1.0);
    float eyeMinRadius = eyeMinSize.x / eyeMinSize.w;

    float eyeRadius =  max(JointRadius, eyeMinRadius);
    float eyeDepth = eye4.z + eyeRadius;
    vec4 clipSize = ProjMat * vec4(eyeRadius, 0.0, eyeDepth, 1.0);
    float viewDepth = ((clipSize.z / clipSize.w) + 1.0) / 2.0;
    float viewRadius = (clipSize.x / clipSize.w) / 2.0;

    size = viewRadius * Viewport.x;
    gl_PointSize = size * 2.0;

    dept = viewDepth - view.z;
}

