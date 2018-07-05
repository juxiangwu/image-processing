/* Please Write the OpenCL Kernel(s) code here*/

#define FXAA_REDUCE_MIN (1.0f / 128.0f)
#define FXAA_REDUCE_MUL (1.0f / 8.0f)
#define FXAA_SPAN_MAX 8.0f

__kernel void fxaa_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 posPos = (float2)(dim.x / 2,dim.y / 2);
   float rtWidth = 0.05f;
   float rtHeight = 0.05f;
   
   float2 coord_src = (float2)(coord.x,coord.y);
   
   coord = coord * (float2)(rtWidth,rtHeight);
   
   float4 color = read_imagef(input,sampler,coord);
   float2 inverseVP = (float2)(1.0f / rtWidth,1.0f / rtHeight);
   
   float3 rgbNW = read_imagef(input,sampler,(coord + (float2)(-1.0f,-1.0f)) * inverseVP).xyz;
   float3 rgbNE = read_imagef(input,sampler,(coord + (float2)(1.0f,-1.0f)) * inverseVP).xyz;
   float3 rgbSW = read_imagef(input,sampler,(coord + (float2)(-1.0f,1.0f)) * inverseVP).xyz;
   float3 rgbSE = read_imagef(input,sampler,(coord + (float2)(-1.0f,1.0f)) * inverseVP).xyz;
   float3 rgbM = read_imagef(input,sampler,coord * inverseVP).xyz;
   
   float3 luma = (float3)(0.299f,0.587f,0.114f);
   float lumaNW = dot(rgbNW,luma);
   float lumaNE = dot(rgbNE,luma);
   float lumaSW = dot(rgbSW,luma);
   float lumaSE = dot(rgbSE,luma);
   float lumaM = dot(rgbM,luma);
   
   float lumaMin = min(lumaM,min(min(lumaNW,lumaNE),min(lumaSW,lumaSE)));
   float lumaMax = max(lumaM,max(max(lumaNW,lumaNE),min(lumaSW,lumaSE)));
   
   float2 dir;
   dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
   dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));
   
   float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25f * FXAA_REDUCE_MUL),FXAA_REDUCE_MIN);
   float rcpDirMin = 1.0f / (min(fabs(dir.x),fabs(dir.y)) + dirReduce);
   float FXAA_SUBPIX_SHIFT = 1.0f / 4.0f;
   dir = min((float2)(FXAA_SPAN_MAX,FXAA_SPAN_MAX),max((float2)(-FXAA_SPAN_MAX,-FXAA_SPAN_MAX),dir * rcpDirMin)) * inverseVP;
   
   float3 color1 = read_imagef(input,sampler,coord * inverseVP + dir * (1.0f / 3.0f - 0.5f)).xyz;
   float3 color2 = read_imagef(input,sampler,coord * inverseVP + dir * (2.0f / 3.0f - 0.5f)).xyz;
   
   float3 rgbA = 0.5f * (color1 + color2);
   
   float3 color3 = read_imagef(input,sampler,coord * inverseVP + dir * -0.5f).xyz;
   float3 color4 = read_imagef(input,sampler,coord * inverseVP + dir * 0.5f).xyz;
   
   float3 rgbB  = rgbA * 0.5f + 0.25f * (color3 + color4);
   
   float lumaB = dot(rgbB,luma);
   
   if(lumaB < lumaMin || lumaB > lumaMax){
       color = (float4)(rgbA,1.0f);
   }else{
       color = (float4)(rgbB,2.0f);
   }
   
   write_imagef(output,convert_int2(coord_src),color);
}