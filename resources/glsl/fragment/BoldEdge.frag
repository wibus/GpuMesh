#version 440

in float dist;


layout(location = 0) out vec4 FragColor;


void main()
{
    if(dist > 0.0)
        discard;

    FragColor = vec4(0, 0, 0, 1);
}
