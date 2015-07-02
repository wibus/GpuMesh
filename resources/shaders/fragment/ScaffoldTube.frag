#version 440

uniform vec3 CameraPosition;
uniform vec3 LightDirection;
uniform float LineRadius;
uniform int LightMode;
uniform int DiffuseLightMode;


in vec3 pos;
flat in vec3 end;
noperspective in vec2 pix;
in float dist;
in float qual;


layout(location = 0) out vec4 FragColor;


const float MAT_SHINE = 40.0;
const vec3 LIGHT_AMBIANT = vec3(0.3);
const vec3 LIGHT_DIFFUSE = vec3(0.5);
const vec3 LIGHT_SPECULAR = vec3(1.0, 0.9, 0.7);


vec3 qualityLut(in float q);

float sphericalDiffuse(in vec3 n);
float lambertDiffuse(in vec3 n);
float phongSpecular(in vec3 n, in vec3 p, in float shine);


void main()
{
    // Cut plane distance
    if(dist > 0.0)
        discard;

    vec2 rad = (gl_FragCoord.xy - pix) / LineRadius;
    float radDist = sqrt(1.0 - min(dot(rad, rad), 1.0));
    float radLength = length(rad) * sign(rad.y);

    vec3 dir = end - pos;
    vec3 cam = normalize(CameraPosition - pos);
    vec3 up = normalize(cross(cam, dir));
    vec3 right = normalize(cross(dir, up));
    if(up.z < 0.0)
        up = -up;

    vec3 normal = normalize(
            radLength * up +
            radDist * right);

    vec3 color;
    if(LightMode == DiffuseLightMode)
    {
        vec3 diff = LIGHT_DIFFUSE * sphericalDiffuse(normal);
        color = qualityLut(qual) * (LIGHT_AMBIANT + diff * 1.5);
    }
    else
    {
        vec3 diff = LIGHT_DIFFUSE * lambertDiffuse(normal);
        vec3 spec = LIGHT_SPECULAR * phongSpecular(normal, pos, MAT_SHINE);
        color = qualityLut(qual) * (LIGHT_AMBIANT + diff) + spec;
    }

    FragColor = vec4(color, 1.0);
}
