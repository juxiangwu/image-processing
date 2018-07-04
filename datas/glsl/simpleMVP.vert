#version 440

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

in vec4 position;
in vec4 colour_in;

out vec4 colour_out;

void main()
{
    gl_Position = proj * view * model * position;
    colour_out = colour_in;
}