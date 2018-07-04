#version 440

in vec4 colour_out;
out vec4 fragColour;

void main()
{
    fragColour = colour_out;
}