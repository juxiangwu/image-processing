/* Please Write the OpenCL Kernel(s) code here*/
__kernel void directional_nonmaximum_suppression_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   int width = size.x;
   int height = size.y;
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float upperThreshold = 0.80;
   float lowerThreshold = 0.25;
   
   float4 currentGradientAndDirection = read_imagef(input,sampler,(coord));
   float2 vUv = (float2)(3,3);
   float2 gradientDirection = ((currentGradientAndDirection.yz * 2.0f) - 1.0f) * (float2)(1.0f/width, 1.0f/height);
   float firstSampledGradientMagnitude = read_imagef(input,sampler,convert_int2(gradientDirection + vUv)).x;
   float secondSampledGradientMagnitude = read_imagef(input,sampler,convert_int2(vUv - gradientDirection)).x;
   float multiplier = step(firstSampledGradientMagnitude, currentGradientAndDirection.x);
   multiplier = multiplier * step(secondSampledGradientMagnitude, currentGradientAndDirection.x);
   float thresholdCompliance = smoothstep(lowerThreshold, upperThreshold, currentGradientAndDirection.x);
   multiplier = multiplier * thresholdCompliance;
   
   write_imagef(output,coord,(float4)(multiplier,multiplier,multiplier,1.0f));
}