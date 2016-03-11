#version 440

uniform vec3 CameraPosition;
uniform vec3 LightDirection;
uniform sampler2D DepthTex;

in vec3 pos;
in vec3 eye;
in vec3 lgt;
in vec3 nrm;
in float qual;
in float dist;

layout(location = 0) out vec4 FragColor;


const float MAT_SHINE = 40.0;
const vec3 LIGHT_DIFF = vec3(0.6);
const vec3 LIGHT_AMBT = vec3(0.3);
const vec3 LIGHT_SPEC = vec3(1.0, 0.9, 0.7);
const float DEPTH_MIN_VAR = 0.05;


vec3 qualityLut(in float q);

float lambertDiffuse(in vec3 n);
float phongSpecular(in vec3 n, in vec3 p, in float shine);


float chebyshevUpperBound(in vec3 l)
{
    vec2 moments = texture(DepthTex, l.xy).rg;

    float p = (l.z <= moments.x) ? 1.0 : 0.0;

    float var = moments.y - (moments.x*moments.x);
    var = max(var, DEPTH_MIN_VAR);

    float d = l.z - moments.x;
    float p_max = var / (var + d*d);

    return max(p, p_max);
}


void main(void)
{
    if(dist > 0.0)
        discard;

    vec3 base = qualityLut(qual);
    float occl = chebyshevUpperBound(lgt);
    vec3 diff = lambertDiffuse(nrm) * LIGHT_DIFF * occl;
    vec3 spec = phongSpecular(nrm, pos, MAT_SHINE) * LIGHT_SPEC * occl;
    FragColor = vec4(base * (LIGHT_AMBT + diff) + spec, 1);
}

