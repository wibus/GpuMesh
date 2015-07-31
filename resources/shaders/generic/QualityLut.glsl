#version 440

vec3 qualityLut(in float q)
{
    vec3 grad = vec3(smoothstep(0.0, 0.15, q) - smoothstep(0.66, 1.0, q),
                     smoothstep(0.5, 0.66, q),
                     smoothstep(-0.15, 0.15, q) - smoothstep(0.15, 0.5, q));
    return mix(grad, vec3(1.0), q == 0);
}
