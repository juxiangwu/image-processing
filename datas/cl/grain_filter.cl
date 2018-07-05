/* Please Write the OpenCL Kernel(s) code here*/
float4 rnm(float2 tc){
    float uTime = 1000.0f;
    float noise = sin(dot(tc + (float2)(uTime,uTime),(float2)(12.9898f,78.233f))) * 43758.5453f;
    
    float iptr;
    float nr = fract(noise,&iptr) * 2.0f - 1.0f;
    float ng = fract(noise * 1.2154f,&iptr) * 2.0f - 1.0f;
    float nb = fract(noise * 1.3453f,&iptr) * 2.0f - 1.0f;
    float na = fract(noise * 1.3647f,&iptr) * 2.0f - 1.0f;
    
    return (float4)(nr,ng,nb,na);
}

float fade(float t){
    return t* t * t * (t * ( t * 6.0f - 15.0f) + 10.0f);
}

float pnoise3D(float3 p){
    float perTexUnit = 1.0f / 256.0f;
    float perTexUnitHalf = 0.5f / 256.0f;
    float3 pi = perTexUnit * floor(p) + perTexUnitHalf;
    float3 iptr;
    float3 pf = fract(p,&iptr);
    
    //noise contributions from (x = 0,y = 0,z = 0 and z = 1)
    float perm00 = rnm(pi.xy).w;
    float3 grad000 = rnm((float2)(perm00,pi.z)).xyz * 4.0f - 1.0f;
    float n000 = dot(grad000,pf);
    float3 grad001 = rnm((float2)(perm00,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n001 = dot(grad001,pf - (float3)(0.0f,0.0f,1.0f));
    
    float perm01 = rnm(pi.xy + (float2)(0.0f,perTexUnit)).w;
    float3 grad010 = rnm((float2)(perm01,pi.z)).xyz * 4.0f - 1.0f;
    float n010 = dot(grad010,pf - (float3)(0.0f,1.0f,0.0f));
    float3 grad011 = rnm((float2)(perm01,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n011 = dot(grad011,pf - (float3)(0.0f,1.0f,1.0f));
    
    float perm10 = rnm(pi.xy + (float2)(perTexUnit,0.0f)).w;
    float3 grad100 = rnm((float2)(perm10,pi.z)).xyz * 4.0f - 1.0f;
    float n100 = dot(grad010,pf - (float3)(1.0f,0.0f,0.0f));
    float3 grad101 = rnm((float2)(perm10,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n101 = dot(grad101,pf - (float3)(1.0f,0.0f,1.0f));
    
    float perm11 = rnm(pi.xy + (float2)(perTexUnit,perTexUnit)).w;
    float3 grad110 = rnm((float2)(perm10,pi.z)).xyz * 4.0f - 1.0f;
    float n110 = dot(grad110,pf - (float3)(1.0f,1.0f,0.0f));
    float3 grad111 = rnm((float2)(perm11,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n111 = dot(grad111,pf - (float3)(1.0f,1.0f,1.0f));
    
    float4 n_x = mix((float4)(n000,n001,n010,n011),(float4)(n100,n101,n110,n111),fade(pf.x));
    
    float2 n_xy = mix(n_x.xy,n_x.zw,fade(pf.y));
    
    float n_xyz = mix(n_xy.x,n_xy.y,fade(pf.z));
    
    return n_xyz;
}

float2 coord_rot(float2 tc,float angle,float aspect){
    float rotX = ((tc.x * 2.0f - 1.0f) * aspect * cos(angle)) - ((tc.y * 2.0f - 1.0f) * sin(angle));
    float rotY = ((tc.y * 2.0f - 1.0f) * cos(angle)) + ((tc.x * 2.0f - 1.0f) * aspect * sin(angle));
    
    rotX = ((rotX / aspect) * 0.5f + 0.5f);
    rotY = rotY * 0.5f + 0.5f;
    
    return (float2)(rotX,rotY);
}

__kernel void grain_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float aspect = (float)dim.x / (float)dim.y;
   
   float3 rotOffset = (float3)(1.425f,3.892f,5.835f);
   float uTime = 1000.0f;
   float grain_size = 1.6f;
   float color_amount = 0.6f;
   float lum_amount = 1.0f;
   float grain_amount = 0.1f;
   float2 rot_coord_R = coord_rot(coord,uTime + rotOffset.x,aspect);
   float3 pr = (float3)(rot_coord_R * (float2)((float)dim.x / grain_size,(float)dim.y / grain_size),0.0f);
   float3 noise = (float3)(pnoise3D(pr));
   
   int colored = 1;
   
   if(colored){
       float2 rot_coord_G = coord_rot(coord,uTime + rotOffset.y,aspect);
       float2 rot_coord_B = coord_rot(coord,uTime + rotOffset.z,aspect);
       float3 pg = (float3)(rot_coord_G * (float2)((float)dim.x / grain_size,(float)dim.y / grain_size),1.0f);
       float3 pb = (float3)(rot_coord_B * (float2)((float)dim.x / grain_size,(float)dim.y / grain_size),2.0f);
       
       noise.y = mix(noise.x,pnoise3D(pg),color_amount);
       noise.z = mix(noise.x,pnoise3D(pb),color_amount);
   }
   
   float3 col = read_imagef(input,sampler,coord).xyz;
   
   float3 lumcoeff = (float3)(0.299f,0.587f,0.114f);
   float luminance = mix(0.0f,dot(col,lumcoeff),lum_amount);
   float lum = smoothstep(0.2f,0.0f,luminance);
   
   lum += luminance;
   
   noise = mix(noise,(float3)(0.0f),pow(lum,0.4f));
   col = col + noise * grain_amount;
   
   write_imagef(output,convert_int2(coord),(float4)(col,1.0f));
}