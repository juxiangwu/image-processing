#version 440

in vec4 position;
in vec4 colour_in;

out vec4 colour_out;

void main()
{
    gl_Position = position;
    colour_out = colour_in;
}