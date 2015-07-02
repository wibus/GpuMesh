#version 440

uniform mat4 ProjMat;
uniform mat4 ViewMat;
uniform vec2 Viewport;
uniform vec4 CutPlaneEq;


layout(location=0) in vec3 position;
layout(location=1) in float quality;


out vec3 pos;
flat out vec3 end;
noperspective out vec2 pix;
out float qual;
out float dist;


void main(void)
{
    pos = position;
    end = position;
    qual = quality;

    dist = dot(position, CutPlaneEq.xyz) - CutPlaneEq.w;

    vec4 pos4 = vec4(position, 1);
    vec4 eye4 = ViewMat * pos4;
    vec4 clip = ProjMat * eye4;
    gl_Position = clip;

    clip = (clip/clip.w + 1.0) / 2.0;
    pix = Viewport * clip.xy;
}
