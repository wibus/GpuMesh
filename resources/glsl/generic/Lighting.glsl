#version 440

uniform vec3 CameraPosition;
uniform vec3 LightDirection;


// Low level shading
float sphericalDiffuse(in vec3 n)
{
    return (dot(n, -LightDirection) + 1.0) / 2.0;
}

float lambertDiffuse(in vec3 n)
{
    return max(dot(n, -LightDirection), 0.0);
}

float phongSpecular(in vec3 n, in vec3 p, in float shine)
{
    vec3 ref = normalize(reflect(p - CameraPosition, n));
    return pow(max(dot(ref, -LightDirection), 0.0), shine);
}
