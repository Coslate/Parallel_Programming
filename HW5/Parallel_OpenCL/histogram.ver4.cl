__kernel void histogram(__global unsigned char *orig_img_d, __global unsigned int *hist_calc_d, const int height, const int width){
    int thread_id_x       = get_global_id(0);
    int thread_id_y       = get_global_id(1);

    int thread_img_index  = (thread_id_y * width + thread_id_x) * 4;

    unsigned int r_val    = orig_img_d[thread_img_index];
    unsigned int g_val    = orig_img_d[thread_img_index + 1] + 256;
    unsigned int b_val    = orig_img_d[thread_img_index + 2] + 512;


    if(thread_id_x < width && thread_id_y < height){
        atomic_inc(&hist_calc_d[r_val]);
        atomic_inc(&hist_calc_d[g_val]);
        atomic_inc(&hist_calc_d[b_val]);
    }
}

__kernel void find_hist_max(__global unsigned int *hist_calc_d){
    int tid       = get_global_id(0);
    atomic_max(&hist_calc_d[768], hist_calc_d[tid]);
}

__kernel void present_result(__global unsigned char *ret_img_d, __global unsigned int *hist_calc_d){
    int thread_id_x       = get_global_id(0);
    int thread_id_y       = get_global_id(1);
    unsigned int max_val  = hist_calc_d[768];
    int r_img_index         = (256 * thread_id_y + thread_id_x)*4;
    int g_img_index       = r_img_index + 1;
    int b_img_index       = r_img_index + 2;
    int a_img_index       = r_img_index + 3;

    if(hist_calc_d[thread_id_x]*256/max_val > thread_id_y)
        ret_img_d[r_img_index] = 255;
    else
        ret_img_d[r_img_index] = 0;
        
    if(hist_calc_d[256 + thread_id_x]*256/max_val > thread_id_y)
        ret_img_d[g_img_index] = 255;
    else
        ret_img_d[g_img_index] = 0;

    if(hist_calc_d[512 + thread_id_x]*256/max_val > thread_id_y)
        ret_img_d[b_img_index] = 255;
    else
        ret_img_d[b_img_index] = 0;

    ret_img_d[a_img_index] = 0;
}

