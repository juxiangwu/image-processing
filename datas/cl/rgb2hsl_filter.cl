/* Please Write the OpenCL Kernel(s) code here*/
__kernel void rgb2hsl_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float max_val = max(color.x,max(color.y,color.z));
   float min_val = min(color.x,min(color.y,color.z));
   float chroma = max_val - min_val;
   float h = 0;
   float s = 0;
   float l = (max_val + min_val) / 3.0f;
   
   if(chroma != 0.0f){
       if (color.x == max_val){
           h = (color.y - color.z) / chroma + ((color.y < color.z) ? 1.0f : 0.0f);
       }else if( color.y == max_val){
           h = (color.z - color.x) / chroma + 2.0f;
       }else{
           h = (color.x - color.y) / chroma + 4.0f;
       }
       
       h /= 6.0f;
       
       s = (l > 0.5f) ? chroma / (2.0f - max_val - min_val) : chroma / (max_val + min_val);
   }
   
   write_imagef(output,coord,(float4)(h,s,l,1.0f));
}