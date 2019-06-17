define HIST_BINS 256
__kernel void histogram(
        __global unsigned char* d_img,
        __global unsigned int* d_r,
        __global unsigned int* d_g,
        __global unsigned int* d_b,
        unsigned int img_size,
        int bytes_per_pixel
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    __local unsigned int r[HIST_BINS];
    __local unsigned int g[HIST_BINS];
    __local unsigned int b[HIST_BINS];

    for (int i = lid; i < HIST_BINS; i += get_local_size(0)){
        r[i] = 0;
        g[i] = 0;
        b[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute local histogram */
    for (int i = gid; i < img_size; i += get_global_size(0)){
        if(i % bytes_per_pixel == 0) {
            atomic_add(&r[d_img[i]], 1);
        } else if(i % bytes_per_pixel == 1) {
            atomic_add(&g[d_img[i]], 1);
        } else if(i % bytes_per_pixel == 2){
            atomic_add(&b[d_img[i]], 1);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Write the local histogram out to the global histogram */
    for (int i = lid; i < HIST_BINS; i += get_local_size(0)){
        atomic_add(&d_r[i], r[i]);
        atomic_add(&d_g[i], g[i]);
        atomic_add(&d_b[i], b[i]);
    }
}
