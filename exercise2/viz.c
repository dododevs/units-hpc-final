#ifdef VIZ
#include <SDL2/SDL.h>
#include <stdio.h>

#include "types.h"
#include "log.h"

#define WINDOW_WIDTH 1024

SDL_Window* window;
SDL_Renderer* renderer;

int viz_init(char* title, int width, int height)
{
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
    return 1;
  }

  float ratio = (float) width / height;
  int win_height = (int) WINDOW_WIDTH / ratio;
  printf("win_height = %d\n", win_height);
  /* window = SDL_CreateWindow( */
  /*   title, */
  /*   SDL_WINDOWPOS_CENTERED, */
  /*   SDL_WINDOWPOS_CENTERED, */
  /*   WINDOW_WIDTH, */
  /*   win_height, */
  /*   SDL_WINDOW_SHOWN | SDL_WINDOW_FULLSCREEN */
  /* ); */
  window = SDL_CreateWindow(
    title,
    SDL_WINDOWPOS_UNDEFINED,
    SDL_WINDOWPOS_UNDEFINED,
    WINDOW_WIDTH,
    win_height,
    SDL_WINDOW_SHOWN
  );

  if (window == NULL) {
    SDL_Log("Unable to create window: %s", SDL_GetError());
    return 1;
  }
  // SDL_SetWindowSize(window, WINDOW_WIDTH, win_height);
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (renderer == NULL) {
    SDL_Log("Unable to create renderer: %s", SDL_GetError());
    return 1;
  }
  // SDL_RenderSetScale(renderer, WINDOW_WIDTH / (float) width, win_height / (float) height);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderClear(renderer);
  SDL_RenderPresent(renderer);
  return 0;
}

void viz_render(mb_t* M, int nx, int ny, int max)
{
  int v;
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      // v = (double) M[y * nx + x] / (double) max * 255;
      v = M[y * nx + x];
      
      SDL_SetRenderDrawColor(renderer, v, v, v, 255);
      SDL_RenderDrawPoint(renderer, x, y);
    }
  }
  SDL_RenderPresent(renderer);
}

void viz_render_pixel(mb_t val, int r, int x, int max)
{
  int v = val * 255 / max;
  SDL_SetRenderDrawColor(renderer, v, v, v, 255);
  SDL_RenderDrawPoint(renderer, x, r);
  SDL_RenderPresent(renderer);
}

void viz_destroy()
{
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
#endif
