#include <stdlib.h>
#include "pgm.h"

void main()
{
  char* image;
  
  image = malloc(sizeof(char) * 4);
  image[0] = 127;
  image[1] = 255;
  image[2] = 127;
  image[3] = 255;

  write_pgm_image(image, 255, 2, 2, "test.pgm");
  free(image);
}