extern "C"{

    __global__ void add_rgb(float3* d_T1, float3* d_T2, float3* d_out, int numRows, int numCols) {
        //get row and column in blcok
        int r = threadIdx.y + blockIdx.y*blockDim.y;
        int c = threadIdx.x + blockIdx.x*blockDim.x;
        //get unique point in image by finding position in grid.
        int index = c + r * numCols;//r*blockDim.x*gridDim.x;
        int totalSize = numRows*numCols;
        if (r < numRows && c < numCols) {
            d_out[index].x = d_T1[index].x + d_T2[index].x;
            d_out[index].y = d_T1[index].x + d_T2[index].y;
            d_out[index].z = d_T1[index].x + d_T2[index].z;
        }
    }

}