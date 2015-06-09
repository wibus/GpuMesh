#version 440

uniform vec3 CameraPosition;

in vec3 pos;
in vec3 nrm;
in vec3 edg;
in float qual;
in float dist;

layout(location = 0) out vec4 FragColor;


vec3 lut(float q)
{
    return vec3(smoothstep(0.0, 0.15, q) - smoothstep(0.66, 1.0, q),
                smoothstep(0.5, 0.66, q),
                smoothstep(-0.15, 0.15, q) - smoothstep(0.15, 0.5, q));
}


vec3 bold(vec3 e)
{
    return vec3(!any(greaterThan(e, vec3(0.99))));
}


const vec3 LIGHT_DIRECTION = normalize(vec3(1, 2, 3));
const vec3 LIGHT_COLOR = vec3(1);
const vec3 LIGHT_DIRECT = vec3(0.55);
const vec3 LIGHT_INDIRECT = vec3(0.25);
const vec3 LIGHT_AMBIANT = vec3(0.2);
vec3 diffuse(in vec3 n)
{
    float inderectLighting = (n.z + 1.0) * LIGHT_INDIRECT;
    float directLighting = max(dot(LIGHT_DIRECTION, n), 0.0) * LIGHT_DIRECT;
    return LIGHT_COLOR * (directLighting + inderectLighting) + LIGHT_AMBIANT;
}

const vec3 LIGHT_SPEC = vec3(1.0, 0.9, 0.7);
const float MAT_SHINE = 40.0;
vec3 specular(in vec3 p, in vec3 n)
{
    vec3 ref = normalize(reflect(p - CameraPosition, n));
    return LIGHT_SPEC * pow(max(dot(ref, LIGHT_DIRECTION), 0.0), MAT_SHINE);
}

void main(void)
{
    if(dist > 0.0)
        discard;

    vec3 baseCol = lut(qual);
    vec3 diffCol = baseCol * bold(edg) * diffuse(nrm);
    vec3 specCol = diffCol + specular(pos, nrm);
    FragColor = vec4(specCol , 1);
}

