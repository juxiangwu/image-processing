#version 440
in layout(location = 0) vec3 position;
in layout(location = 1) vec2 textureCoords;

out vec2 textures;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0f);
    textures = vec2(textureCoords.x, 1 - textureCoords.y);
}
