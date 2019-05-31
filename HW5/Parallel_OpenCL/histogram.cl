__kernel void histogram(__global unsigned char *orig_img_d, __global unsigned int *hist_calc_d, const int height, const int width){
    int thread_id_x       = get_global_id(0);
    int thread_id_y       = get_global_id(1);
    int thread_id_z       = get_global_id(2);

    if(thread_id_x < width && thread_id_y < height){
        int color_offset    = thread_id_z << 8;
        atomic_inc(&hist_calc_d[color_offset + orig_img_d[thread_id_y * width + thread_id_x + thread_id_z]]);
    }
}
