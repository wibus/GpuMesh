#version 440

in float dist;

layout(location = 0) out vec4 FragColor;

void main(void)
{
    if(dist > 0.0)
        discard;

    float depth = gl_FragCoord.z;
    float variance = depth * depth;
    FragColor = vec4(depth, variance, 0, 1.0);
}

