#include <stdarg.h>
#include <stdio.h>

#ifndef LOG_LEVEL
#define LOG_LEVEL 4
#endif

void llog(int level, const char *format, ...) {
  if (level > LOG_LEVEL) {
    return;
  }
  va_list args;
  va_start(args, format);
  vprintf(format, args);    
  va_end(args);
}