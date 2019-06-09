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
