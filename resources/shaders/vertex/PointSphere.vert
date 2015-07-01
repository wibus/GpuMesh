#version 440

uniform mat4 ProjMat;
uniform mat4 ViewMat;
uniform vec2 Viewport;
uniform vec4 CutPlaneEq;
uniform float PointRadius;
uniform float LineWidth;

layout(location=0) in vec3 position;
layout(location=1) in float quality;

out vec3 pos;
noperspective out vec2 pix;
out float qual;
out float size;
out float dept;
out float dist;


void main(void)
{
    pos = position;
    qual = quality;

    dist = dot(position, CutPlaneEq.xyz) - CutPlaneEq.w;


    vec4 pos4 = vec4(position, 1);
    vec4 eye4 = ViewMat * pos4;
    vec4 clip = ProjMat * eye4;
    gl_Position = clip;

    clip = (clip/clip.w + 1.0) / 2.0;
    pix = Viewport * clip.xy;


    vec4 clipPoint = ProjMat * vec4(PointRadius*2.0, 0.0, eye4.z + PointRadius, 1.0);
    clipPoint = (clipPoint / clipPoint.w + 1.0) / 2.0;
    gl_PointSize = max(Viewport.x * (clipPoint.x-0.5), LineWidth * 1.5);

    size = gl_PointSize / 2.0;
    dept = clipPoint.z - clip.z;
}

