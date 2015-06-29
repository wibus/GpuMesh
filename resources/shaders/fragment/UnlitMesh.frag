#version 440

uniform vec3 CameraPosition;
uniform vec3 LightDirection;

in vec3 pos;
in vec3 eye;
in vec3 nrm;
in vec3 edg;
in float qual;
in float dist;

layout(location = 0) out vec4 FragColor;


const vec3 LIGHT_DIFFUSE = vec3(0.6);
const vec3 LIGHT_AMBIANT = vec3(0.3);


vec3 lut(in float q)
{
    return vec3(smoothstep(0.0, 0.15, q) - smoothstep(0.66, 1.0, q),
                smoothstep(0.5, 0.66, q),
                smoothstep(-0.15, 0.15, q) - smoothstep(0.15, 0.5, q));
}

vec3 bold(in vec3 e, in float d)
{
    const float THRESHOLD = 0.02;
    float smoothWidth = 0.04 * d;
    float inf = THRESHOLD - smoothWidth;
    float sup = THRESHOLD + smoothWidth;

    float maxE = max(max(e.x, e.y), e.z);
    return vec3(smoothstep(inf, sup, 1.0 - maxE));
}

vec3 diffuse(in vec3 n)
{
    return  LIGHT_DIFFUSE * max(dot(-LightDirection, n), 0.0);
}

void main(void)
{
    if(dist > 0.0)
        discard;

    float dist = length(eye);
    vec3 baseCol = lut(qual) * bold(edg, dist);
    vec3 diffCol = diffuse(nrm) + LIGHT_AMBIANT;
    FragColor = vec4(baseCol * diffCol, 1);
}

