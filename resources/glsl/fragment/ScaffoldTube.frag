#version 440

uniform float TubeRadius;
uniform float JointTubeMinRatio;
uniform vec3 CameraPosition;
uniform vec3 LightDirection;
uniform int DiffuseLightMode;
uniform int LightMode;

in vec3 wPos;
flat in vec3 wEnd;
flat in vec2 pEnd;
noperspective in vec2 pPos;
in float dept;
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

    vec2 pixRad = (gl_FragCoord.xy - pPos) / TubeRadius;
    float radDist = sqrt(1.0 - min(dot(pixRad, pixRad), 1.0));
    vec2 pixUp = pEnd - pPos; pixUp = vec2(-pixUp.y, pixUp.x);
    float sideDist = length(pixRad) * sign(dot(pixRad, pixUp));

    vec3 dir = normalize(wEnd - wPos);
    vec3 cam = normalize(CameraPosition - wPos);
    vec3 side = normalize(cross(cam, dir));
    vec3 right = normalize(cross(dir, side));

    vec3 normal = normalize(
            sideDist * side +
            radDist * right);

    vec3 color;
    if(LightMode == DiffuseLightMode)
    {
        vec3 diff = LIGHT_DIFFUSE * sphericalDiffuse(normal);
        color = qualityLut(qual) * (LIGHT_AMBIANT + diff * 1.25);
    }
    else
    {
        vec3 diff = LIGHT_DIFFUSE * lambertDiffuse(normal);
        vec3 spec = LIGHT_SPECULAR * phongSpecular(normal, wPos, MAT_SHINE);
        color = qualityLut(qual) * (LIGHT_AMBIANT + diff) + spec;
    }


    float camDirCos = abs(dot(cam, dir));
    float camDirSin = sqrt(1.0 - camDirCos * camDirCos);
    float deptCoeff = min(radDist / camDirSin, JointTubeMinRatio * (9.0/10.0));
    gl_FragDepth = gl_FragCoord.z + dept * deptCoeff;


    FragColor = vec4(color, 1.0);
}
