#version 440

vec3 qualityLut(in float q)
{
    q = clamp(q, 0.0, 1.0);

    vec3 grad = vec3(
        // Red
        clamp(2.0 - 4*q, 0.0, 1.0),
        // Green

        clamp(1.3*(1.0 - pow(2*abs(0.5-q), 2)), 0.0, 1.0),

        // Blue
        clamp(4*q - 2, 0.0, 1.0));

    //grad /= max(max(grad.r, grad.g), grad.b);

    return mix(grad, vec3(1.0), q <= 0);
}
