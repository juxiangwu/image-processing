/* Please Write the OpenCL Kernel(s) code here*/
__kernel void below_ab_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float thresh_a = 0.5f;
   float thresh_b = 0.5f;
   
   
   
   float4 color = read_imagef(input,sampler,coord);
   
   
   if(color.x > thresh_a){
       float4 rgba = (float4)(clamp((color.x - thresh_a) / (1.0f - thresh_a),0.0f,1.0f),0.0f,0.0f,1.0f);
       
   }else{
       if(color.x > thresh_b){
           float4 rgba = (float4)(0.0f,clamp((color.y - thresh_b) / (1.0f - thresh_b),0.0f,1.0f),0.0f,1.0f);
           write_imagef(output,convert_int2(coord),rgba);
       }else{
           float4 rgba = (float4)(0.0,0.0,color.y / thresh_b,1.0f);
           write_imagef(output,convert_int2(coord),rgba);
       }
   }
}