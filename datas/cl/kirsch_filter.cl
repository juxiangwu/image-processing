/* Please Write the OpenCL Kernel(s) code here*/

__kernel void kirsch_filter(__read_only image2d_t input, __write_only image2d_t output){

    const int kirsh_mask [] = {
    5,5,5,-3,0,-3,-3,-3,-3,
    -3,5,5,-3,0,5,-3,-3,-3,
    -3,-3,5,-3,0,5,-3,-3,5,
    -3,-3,-3,-3,0,5,-3,5,5,
    -3,-3,-3,-3,0,-3,5,5,5,
    -3,-3,-3,5,0,-3,5,5,-3,
    5,-3,-3,5,0,-3,5,-3,-3,
    5,5,-3,5,0,-3,-3,-3,-3
    };

    const sampler_t sampler = CLK_FILTER_NEAREST |
                         CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    const int maskwidth = 3;
    const int maskheight = 3;
    const int maskrows = 8 / 2;
    const int maskcols = 9 / 2;
    float4 sum = (float4)(0,0,0,0);
    float4 color = (float4)(0,0,0,0);
    int idx = 0;
    
    for(int i = -maskrows;i <= maskrows;i++){
        for(int j = -maskcols;j <= maskcols;j++){
            sum += read_imagef(input,sampler,coord + (int2)(j,i)) * kirsh_mask[idx];
            idx++;
        }
    }
    
    sum = clamp(sum,0.0f,1.0f);
    write_imagef(output,coord,sum);

}