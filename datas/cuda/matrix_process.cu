Matrix* copy(Matrix* src, Matrix* dest)
{
    int n = src->rows * src->cols;
    int threads = threads_per_block(n);
    int blocks = num_blocks(n, threads);
    do_copy<<<blocks,threads>>>(src, dest);
    cudaDeviceSynchronize();
    return dest;
}

__global__ void do_copy(Matrix* src, Matrix* dest)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < src->rows * src->stride)
    {
        dest->data[index] = src->data[index];
    }
}

// CONVOLUTION

Matrix* convolve(Matrix* M, CONV_KERNEL_TYPE ck_type, Matrix* custom_kernel,
                 BORDER_TYPE b, float* bargs, int anchor_x, int anchor_y)
{
    Matrix* dest = init_matrix(M->rows, M->cols);
    int n = M->rows * M->cols;
    int threads = threads_per_block(n);
    Matrix* ck;
    switch (ck_type)
    {
        case BOX:
            ck = scalar(ones(3, 3), 1.0 / 9);
            break;
        case CUSTOM:
            ck = custom_kernel;
        default:
            // printf("no kernel to convolve with!\n");
            return NULL;
    }
    // printf("convolving with kernel: \n");
    print_matrix(ck);
    do_convolution<<<num_blocks(n, threads),threads>>>(M, ck, dest, b, bargs, anchor_x, anchor_y);
    cudaDeviceSynchronize();
    free_matrix(ck);
    free_matrix(M);
    return dest;
}


__global__
void do_convolution(Matrix* M, Matrix* ck, Matrix* dest, BORDER_TYPE b, float* bargs, int anchor_x, int anchor_y)
{
    int target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target < M->rows * M->stride)
    {
        // printf("target = %d\n", target);
        // anchor offset
        int anchor_index = target - anchor_y * M->stride - anchor_y;
        // printf("anchor_index = %d\n", anchor_index);
        // calculate row and column for edge checking
        int roi_row = anchor_index / M->stride - anchor_y - ck->rows / 2;
        int roi_col = anchor_index % M->stride - anchor_x - ck->cols / 2;
        // printf("roi_row, roi_col = %d, %d\n", roi_row, roi_col);
        int ck_index = 0;
        float sum = 0;
        for (int row = roi_row; row < roi_row + ck->rows ; row++)
        {
            // printf("    row = %d\n", row);
            for (int col = roi_col; col < roi_col + ck->cols; col++)
            {
                // printf("        col = %d\n", col);
                float val, prod;
                val = ((col < 0) || (col >= M->cols) ||
                       (row < 0) || (row >= M->rows)) ?
                    border_val(M, row * M->stride + col, b, bargs) :
                    M->data[row * M->stride + col];
                // printf("            val = %f\n", val);
                prod = val * ck->data[ck_index++];
                // printf("            prod = %f\n", prod);
                sum += prod;
                // printf("            sum = %f\n", sum);
            }
        }
        __syncthreads();
        dest->data[target] = sum;
        // printf("result: %f\n", dest->data[target]);
    }

}

__global__
void do_on_submatrix(Matrix* M, int height, int width, int start_index)
{
    Matrix s = submatrix(M, height, width, start_index);
    for (int i = 0; i < s.rows; i++)
    {
        for (int j = 0; j < s.cols; j++)
        {
            s.data[i*s.stride + j] = 2.0;
        }
    }
}

__device__
Matrix submatrix(Matrix* M, int height, int width, int start_index)
{
    Matrix N;
    N.rows = height;
    N.cols = width;
    N.stride = M->stride;
    N.start = start_index;
    N.data = (N.start >= 0) ? &M->data[start_index] : &M->data[0];
    return N;
}

// TRANSFORMS
Matrix* translate(Matrix* M, int dx, int dy, int bg_value)
{
    Matrix* t_mat = identity_matrix(3);
    t_mat->data[2] = dx;
    t_mat->data[5] = dy;
    return affine_transform(M, t_mat, bg_value);
}

Matrix* rotate(Matrix* M, float radians, int origin_x, int origin_y,
               int bg_value)
{
    if (origin_x < 0)
    {
        origin_x = M->cols / 2;
    }
    if (origin_y < 0)
    {
        origin_y = M->rows / 2;
    }
    Matrix* r_mat = identity_matrix(3);
    r_mat->data[0] = cosf(radians);
    r_mat->data[1] = sinf(radians) * -1;
    r_mat->data[3] = sinf(radians);
    r_mat->data[4] = cosf(radians);
    return affine_transform(M, r_mat, origin_x, origin_y);
}

Matrix* affine_transform(Matrix* M, Matrix* t_mat, int origin_x, int origin_y, int bg_value)
{
    Matrix* m_trans = init_matrix(M->rows, M->cols, bg_value);
    printf("transformation matrix:\n");
    print_matrix(t_mat);
    int n = M->rows * M->cols;
    int threads = threads_per_block(n);
    do_affine_transform<<<num_blocks(n, threads), threads>>>
        (M, t_mat, m_trans, origin_x, origin_y);
    cudaDeviceSynchronize();
    free_matrix(t_mat);
    free_matrix(M);
    return m_trans;
}

// t_mat assumed to be 3x3 transformation matrix
__global__
void do_affine_transform(Matrix* M, Matrix* t_mat, Matrix* dest,
                         int origin_x, int origin_y)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int x = index % M->stride;
    int y = index / M->stride;
    if ((x >= 0) && (y >= 0) &&
        (x < M->cols) && (y < M->rows))
    {
        x -= origin_x;
        y -= origin_y;
        float xp = x * t_mat->data[0] + y * t_mat->data[1] + t_mat->data[2];
        float yp = x * t_mat->data[3] + y * t_mat->data[4] + t_mat->data[5];
        xp += origin_x;
        yp += origin_y;        
        int nx = floor(xp);
        int ny = floor(yp);
        int ni = ny * M->stride + nx;

        if ((nx >= 0) && (ny >= 0) &&
            (nx < M->cols) && (ny < M->rows))
        {
            if ((nx == xp) && (ny == yp)) // don't need interpolate
            {
                dest->data[index] = M->data[ni];
            }
            else if ((nx < M->cols - 1) && (ny < M->rows - 1))
            {
                float dx = xp - nx;
                float dy = yp - ny;
                dest->data[index] = (1 - dx) * (1 - dy) * M->data[ni] +
                    dx * (1 - dy) *  M->data[ni + 1] +
                    (1 - dx) * dy *  M->data[ni + M->stride] +
                    dx * dy *  M->data[ni + M->stride + 1];
            }
        }
    }
}

// HELPERS
__device__
float border_val(Matrix* M, int target_index, BORDER_TYPE b, float* args)
{
    switch (b)
    {
    case BLACK: return 0;
    case WHITE: return 255;
    case VALUE: return args[0];
    default: return 0;
    }
}