/* Please Write the OpenCL Kernel(s) code here*/
#define PI_F 3.14159265358979323846f
float lerp(float t,float a,float b){
    return a + t * ( b - a);
}

float float_mod(float a,float b){
    int n = (int)a / (int)b;
    
    float aa = a - (float)n * b;
    if(aa < 0){
        return a + b;
    }else{
        return aa;
    }
}

__kernel void flare_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float rays = 50.0f;
   float radius = 50.0f;
   float baseAmount = 1.0f;
   float ringAmount = 0.2f;
   float rayAmount = 0.1f;
   
   float centerX = 0.5f,centerY = 0.5f;
   float ringWidth = 1.6f;
   
   float linear = 0.03f;
   float gauss = 0.006f;
   float mix_val = 0.50f;
   float falloff = 6.0f;
   float sigma = radius / 3.0f;
   
   float iCenterX = centerX * dim.x;
   float iCenterY = centerY * dim.y;
   
   float dx = coord.x - iCenterX;
   float dy = coord.y - iCenterY;
   
   float dist = sqrt(dx * dx + dy * dy);
   float a = exp(-dist * dist * gauss) * mix_val + exp(-dist * linear) * (1 - mix_val);
   
   float ring;
   
   a *= baseAmount;
   
   if(dist > radius + ringWidth){
       a = lerp((dist - (radius + ringWidth)) / falloff,a,0);
   }
   
   if(dist < radius - ringWidth || dist > radius + ringWidth){
       ring = 0;
   }else{
       ring = fabs(dist - radius) / ringWidth;
       ring = 1.0f - ring * ring * (3.0f - 2.0f * ring);
       ring *= ringAmount;
   }
   
   a += ring;
   
   float angle = atan2(dx,dy) + PI_F;
   
   angle = (float_mod(angle / PI_F * 17.0f + 1.0f + rand(coord.x * angle),1.0f) - 0.5f) * 2.0f;
   angle = fabs(angle);
   angle = pow(angle,5.0f);
   
   float b = rayAmount * angle / (1.0f + dist * 0.1f);
   a += b;
   
   a = clamp(a,0.0f,1.0f);
   
   float4 mask_color = (float4)(1.0f,0.5f,0.5f,1.0f);
   float4 color = read_imagef(input,sampler,coord);
   float4 rgba;
   rgba.x = lerp(a,color.x,mask_color.x);
   rgba.y = lerp(a,color.y,mask_color.y);
   rgba.z = lerp(a,color.z,mask_color.z);
   rgba.w = 1.0f;
   write_imagef(output,convert_int2(coord),rgba);
}