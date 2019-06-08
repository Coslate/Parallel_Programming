__kernel void histogram(__global unsigned char *orig_img_d, __global unsigned int *hist_calc_d, const int height, const int width){
    int thread_id_x       = get_global_id(0);
    int thread_id_y       = get_global_id(1);

    if(thread_id_x < width && thread_id_y < height){
        atomic_inc(&hist_calc_d[orig_img_d[(thread_id_y * width + thread_id_x)*4]]);
        atomic_inc(&hist_calc_d[256 + orig_img_d[(thread_id_y * width + thread_id_x)*4 + 1]]);
        atomic_inc(&hist_calc_d[512 + orig_img_d[(thread_id_y * width + thread_id_x)*4 + 2]]);
    }
}
