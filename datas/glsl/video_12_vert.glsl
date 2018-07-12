#version 440
in layout(location = 0) vec3 position;
in layout(location = 1) vec2 textureCoords;
in layout(location = 2) vec3 normals;


out vec2 textures;
out vec3 fragNormal;

uniform mat4 light;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 rotate;

void main()
{
    fragNormal = (light * vec4(normals, 0.0f)).xyz;
    gl_Position = projection * view * model * rotate * vec4(position, 1.0f);
    textures = vec2(textureCoords.x, 1 - textureCoords.y);
}
