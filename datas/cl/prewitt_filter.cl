/* Please Write the OpenCL Kernel(s) code here*/
__kernel void prewitt_filter(__read_only image2d_t input, __write_only image2d_t output){

    const int prewitt_mask_h [9] = {1,1,1,0,0,0,-1,-1,-1};
    const int prewitt_mask_v [9] = {-1,0,1,-1,0,1,-1,0,1};

    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));
     const int maskSize = 3;
     const int maskrows = maskSize / 2;
     const int maskcols = maskSize / 2;

     float4 color = (float4)(0,0,0,0);
     float4 colorv = (float4)(0,0,0,0);
     float4 colorh = (float4)(0,0,0,0);

     int mask_idx = 0;
     for(int y = -maskrows;y <= maskrows;++y){

         for(int x = -maskcols; x <= maskcols;++x){
             float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));
             
             colorh += srcColor * prewitt_mask_h[mask_idx];
            

             colorv += srcColor * prewitt_mask_v[mask_idx];
            
             color.x = colorh.x > colorv.x ? colorh.x : colorv.x;
             color.y = colorh.y > colorv.y ? colorh.y : colorv.y;
             color.z = colorh.z > colorv.z ? colorh.z : colorv.z;


             mask_idx += 1;
         }
     }

     write_imagef(output,coord,color);
}