#version 440

uniform vec3 CameraPosition;
uniform vec3 LightDirection;
uniform sampler2D DepthTex;

in vec3 pos;
in vec3 lgt;
in vec3 nrm;
in vec3 edg;
in float qual;
in float dist;

layout(location = 0) out vec4 FragColor;


const float MAT_SHINE = 40.0;
const vec3 LIGHT_COLOR = vec3(1);
const vec3 LIGHT_DIRECT = vec3(0.55);
const vec3 LIGHT_AMBIANT = vec3(0.2);
const vec3 LIGHT_SPEC = vec3(1.0, 0.9, 0.7);
const float DEPTH_MIN_VAR = 0.05;


vec3 lut(in float q)
{
    return vec3(smoothstep(0.0, 0.15, q) - smoothstep(0.66, 1.0, q),
                smoothstep(0.5, 0.66, q),
                smoothstep(-0.15, 0.15, q) - smoothstep(0.15, 0.5, q));
}

vec3 bold(in vec3 e)
{
    return vec3(!any(greaterThan(e, vec3(0.99))));
}

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

vec3 diffuse(in vec3 n)
{
    float directLighting = max(dot(-LightDirection, n), 0.0) * LIGHT_DIRECT;
    return LIGHT_COLOR * directLighting;
}

vec3 specular(in vec3 p, in vec3 n)
{
    vec3 ref = normalize(reflect(p - CameraPosition, n));
    return LIGHT_SPEC * pow(max(dot(ref, -LightDirection), 0.0), MAT_SHINE);
}

void main(void)
{
    if(dist > 0.0)
        discard;

    float occl = chebyshevUpperBound(lgt);
    vec3 baseCol = lut(qual) * bold(edg);
    vec3 diffCol = diffuse(nrm) * occl + LIGHT_AMBIANT;
    vec3 specCol = specular(pos, nrm) * occl;
    FragColor = vec4(baseCol * diffCol + specCol, 1);
}

