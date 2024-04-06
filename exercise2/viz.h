int viz_init(char* title, int width, int height);
void viz_render(void* M, int nx, int ny, int max);
void viz_render_pixel(int val, int r, int x, int max);
void viz_destroy();