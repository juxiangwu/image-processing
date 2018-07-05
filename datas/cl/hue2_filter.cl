/* Please Write the OpenCL Kernel(s) code here*/

__kernel void hue2_filter(__read_only image2d_t input,
                         __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   const float4 kRGBToYPrime = (float4) (0.299, 0.587, 0.114, 0.0);
   const float4 kRGBToI     = (float4) (0.595716, -0.274453, -0.321263, 0.0);    
   const float4 kRGBToQ     = (float4) (0.211456, -0.522591, 0.31135, 0.0);
 
   const float4 kYIQToR   = (float4) (1.0, 0.9563, 0.6210, 0.0);
   const float4 kYIQToG   = (float4) (1.0, -0.2721, -0.6474, 0.0);
   const float4 kYIQToB   = (float4) (1.0, -1.1070, 1.7046, 0.0);
   
   float4 color = read_imagef(input,sampler,coord);
   
   float YPrime  = dot (color, kRGBToYPrime);
   float I      = dot (color, kRGBToI);
   float Q      = dot (color, kRGBToQ);
   
   float hue     = atan2 (Q, I);
   float chroma  = sqrt (I * I + Q * Q);
   
   float hueAdjust = 0.0f;
   
   hue += (-hueAdjust);
   
   Q = chroma * sin (hue);
   I = chroma * cos (hue);
     
     // Convert back to RGB
   float4 yIQ = (float4) (YPrime, I, Q, 0.0);
   color.x = dot (yIQ, kYIQToR);
   color.y = dot (yIQ, kYIQToG);
   color.z = dot (yIQ, kYIQToB);
   
   write_imagef(output,coord,color);
}