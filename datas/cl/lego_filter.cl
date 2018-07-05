/* Please Write the OpenCL Kernel(s) code here*/

__kernel void lego_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,convert_int2(coord));
   
   float arg = 0.75f;
   
   if(arg > 0.0f){
    float xBlockSize = arg * 0.1f;
    float yBlockSize = xBlockSize * size.x / size.y;  // mutiply ratio
    float xCoord = (floor((coord.x - 0.5) / xBlockSize) + 0.5f) * xBlockSize + 0.5f;
    float yCoord = (floor((coord.y - 0.5)/ yBlockSize) + 0.5f) * yBlockSize + 0.5f;
    float4 rgba = read_imagef(input,sampler, convert_int2((float2)(xCoord, yCoord)));
    float sum = (rgba.x + rgba.y + rgba.z) / 3.0f;
    float3 one = (float3)(255.0f, 255.0f, 255.0f) / 255.0f;
    float3 two = (float3)(242.0f, 252.0f, 0.0f) / 255.0f;
    float3 three = (float3)(0.0f, 140.0f, 0.0f) / 255.0f;
    float3 four = (float3)(48.0f, 19.0f, 6.0f) / 255.0f;
    float3 five = (float3)(0.0f, 0.0f, 0.0f) / 255.0f;
/*
ü\¡¡¾v¡¡»ÆÉ«¡¡°×
1   255 255 255
2   242 252 0
3   0   140 0
4   48  19  6
5   0   0   0
*/
    if      (sum < 0.05){ 
      rgba = (float4)(five,   1.0f);
    }
    else if (sum < 0.65) {
      rgba = (float4)(four,   1.0f);
    }
    else if (sum < 1.40) {
      rgba = (float4)(three, 1.0f);
    }
    else if (sum < 2.15) {
     rgba = (float4)(two,  1.0f);
    }
    else{                 
       rgba = (float4)(one,  1.0f);
    }
   // rgba = color;
    write_imagef(output,convert_int2(coord),rgba);
   }else{
       write_imagef(output,convert_int2(coord),color);
   }
   
}
