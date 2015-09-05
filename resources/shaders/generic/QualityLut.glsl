#version 440

vec3 qualityLut(in float q)
{
    q = clamp(q, 0.0, 1.0);

    vec3 grad = vec3(
        // Red
        1.0 - smoothstep(0.25, 0.5, q),
        // Green
        smoothstep(0.0, 0.25, q) - smoothstep(0.75, 1.0, q),
        // Blue
        smoothstep(0.5, 0.75, q));

    return mix(grad, vec3(1.0), q <= 0);
}
