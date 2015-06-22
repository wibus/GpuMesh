#version 440

uniform vec3 CameraPosition;
uniform vec3 LightDirection;
uniform sampler2D DepthTex;

in vec3 pos;
in vec3 eye;
in vec3 lgt;
in vec3 nrm;
in vec3 edg;
in float qual;
in float dist;

layout(location = 0) out vec4 FragColor;


const float MAT_SHINE = 40.0;
const vec3 LIGHT_DIFFUSE = vec3(0.6);
const vec3 LIGHT_AMBIANT = vec3(0.3);
const vec3 LIGHT_SPEC = vec3(1.0, 0.9, 0.7);
const float DEPTH_MIN_VAR = 0.05;


vec3 lut(in float q)
{
    return vec3(smoothstep(0.0, 0.15, q) - smoothstep(0.66, 1.0, q),
                smoothstep(0.5, 0.66, q),
                smoothstep(-0.15, 0.15, q) - smoothstep(0.15, 0.5, q));
}

vec3 bold(in vec3 e, in float d)
{
    const float THRESHOLD = 0.01;
    float smoothWidth = THRESHOLD * 1.5 * d;
    float inf = max(THRESHOLD - smoothWidth, -THRESHOLD);
    float sup = THRESHOLD + smoothWidth;

    float maxE = max(max(e.x, e.y), e.z);
    return vec3(smoothstep(inf, sup, 1.0 - maxE));
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
    return  LIGHT_DIFFUSE * max(dot(-LightDirection, n), 0.0);
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

    float dist = sqrt(length(eye));
    float occl = chebyshevUpperBound(lgt);
    vec3 baseCol = lut(qual) * bold(edg, dist);
    vec3 diffCol = diffuse(nrm) * occl + LIGHT_AMBIANT;
    vec3 specCol = specular(pos, nrm) * occl;
    FragColor = vec4(baseCol * diffCol + specCol, 1);
}

