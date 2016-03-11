#version 440

uniform vec3 CameraPosition;
uniform vec3 LightDirection;

in vec3 pos;
in vec3 eye;
in vec3 nrm;
in float qual;
in float dist;

layout(location = 0) out vec4 FragColor;


const vec3 LIGHT_DIFFUSE = vec3(0.6);
const vec3 LIGHT_AMBIANT = vec3(0.3);


vec3 qualityLut(in float q);

float lambertDiffuse(in vec3 n);


vec3 diffuse(in vec3 n)
{
    return  LIGHT_DIFFUSE * max(dot(-LightDirection, n), 0.0);
}

void main(void)
{
    if(dist > 0.0)
        discard;

    vec3 baseCol = qualityLut(qual);
    vec3 diffCol = lambertDiffuse(nrm) * LIGHT_DIFFUSE;
    FragColor = vec4(baseCol * (LIGHT_AMBIANT + diffCol), 1);
}

