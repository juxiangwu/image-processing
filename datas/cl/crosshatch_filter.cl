/* Please Write the OpenCL Kernel(s) code here*/
__kernel void crosshatch_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   float lum = length(color.xyz);
   write_imagef(output,convert_int2(coord),color);
   if(lum < 1.0f){
       if(fmod(coord.x + coord.y,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
   
   if(lum < 0.75f){
        if(fmod(coord.x - coord.y,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
   
   if(lum < 0.50f){
        if(fmod(coord.x + coord.y - 5.0f,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
   
   if(lum < 0.3f){
        if(fmod(coord.x - coord.y - 5.0f,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
}