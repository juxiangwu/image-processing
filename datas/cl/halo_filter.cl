/* Please Write the OpenCL Kernel(s) code here*/
__kernel void halo_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   
   
   float gauss [] = {1,2,1,2,4,2,1,2,1};
   float r = 150.0f * 150.0f;
   float x = 150.0;
   float y = 150.0f;
   float delta = 48.0f;
   
   
   float dist = pow(coord.x - x,2.0f) + pow(coord.y - y,2.0f);
   int idx = 0;
   if(dist > r){
       float3 rgb = (float3)(0,0,0);
       for(int m = -1; m <= 1;m++){
           for(int n = -1;n <= 1;n++){
               float4 src_rgba = read_imagef(input,sampler,coord + (int2)(m,n)) * 255.0f;
               rgb.x = rgb.x + src_rgba.x * gauss[idx];
               rgb.y = rgb.y + src_rgba.y * gauss[idx];
               rgb.z = rgb.z + src_rgba.z * gauss[idx];
               idx++;
           }
       }
       
       rgb /= delta;
       
       if(rgb.x < 0){
           rgb.x = -rgb.x;
       }
       
       if(rgb.x > 255){
           rgb.x = 255;
       }
       
        if(rgb.y < 0){
           rgb.y = -rgb.y;
       }
       
       if(rgb.y > 255){
           rgb.y = 255;
       }
       
        if(rgb.z < 0){
           rgb.z = -rgb.z;
       }
       
       if(rgb.z > 255){
           rgb.z = 255;
       }
      
       rgb /= 255.0f;
       
       write_imagef(output,coord,(float4)(rgb,1.0f));
   }else{
       write_imagef(output,coord,color / 255.0f);
   }
}