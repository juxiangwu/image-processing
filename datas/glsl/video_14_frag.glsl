#version 440
in vec2 textures;

out vec4 outColor;
uniform sampler2D sampTexture;

void main()
{
    outColor = texture(sampTexture, textures);
}
