#version 440
in vec2 textures;
in vec3 fragNormal;

out vec4 outColor;
uniform sampler2D sampTexture;

void main()
{
    vec3 ambientLightIntensity = vec3(0.3f, 0.2f, 0.4f);
    vec3 sunLightIntensity = vec3(0.9f, 0.9f, 0.9f);
    vec3 sunLightDirection = normalize(vec3(1.0f, 1.0f, -0.5f));

    vec4 texel = texture(sampTexture, textures);
    vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(fragNormal, sunLightDirection), 0.0f);
    outColor = vec4(texel.rgb * lightIntensity, texel.a);
}
