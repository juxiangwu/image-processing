/* Please Write the OpenCL Kernel(s) code here*/

__kernel void light_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   int width = dim.x;
   int height = dim.y;
   
   int halfw = width / 2;
   int halfh = height / 2;
   
   int R = min(halfw,halfh);
   float light = 150.0f / 255.0f;
   float len = sqrt(pow(coord.x - (float)halfw,2.0f) + pow(coord.y - (float)halfh,2.0f));
   if(len < R){
       float pixel = light * (1.0f - len / R);
       color.x = color.x + pixel;//min(0.0f,min(color.x + pixel,1.0f));
       color.y = color.y + pixel;//min(0.0f,min(color.y + pixel,1.0f));
       color.z = color.z + pixel;//min(0.0f,min(color.z + pixel,1.0f));
   }
   
   write_imagef(output,convert_int2(coord),color);
   
}