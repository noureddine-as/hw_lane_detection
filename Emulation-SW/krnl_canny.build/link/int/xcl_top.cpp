#include "libspir_types.h"
#include "hls_stream.h"
#include "xcl_top_defines.h"
#include "ap_axi_sdata.h"
#define EXPORT_PIPE_SYMBOLS 1
#include "cpu_pipes.h"
#undef EXPORT_PIPE_SYMBOLS
#include "xcl_half.h"
#include <cstddef>
#include <vector>
#include <complex>
#include <pthread.h>
using namespace std;

extern "C" {

void houghlines_accel(size_t img_in, unsigned int threshold, unsigned int maxlines, size_t arrayy, size_t arrayx, unsigned int rows, unsigned int cols);

static pthread_mutex_t __xlnx_cl_houghlines_accel_mutex = PTHREAD_MUTEX_INITIALIZER;
void __stub____xlnx_cl_houghlines_accel(char **argv) {
  void **args = (void **)argv;
  size_t img_in = *((size_t*)args[0+1]);
  unsigned int threshold = *((unsigned int*)args[1+1]);
  unsigned int maxlines = *((unsigned int*)args[2+1]);
  size_t arrayy = *((size_t*)args[3+1]);
  size_t arrayx = *((size_t*)args[4+1]);
  unsigned int rows = *((unsigned int*)args[5+1]);
  unsigned int cols = *((unsigned int*)args[6+1]);
 pthread_mutex_lock(&__xlnx_cl_houghlines_accel_mutex);
  houghlines_accel(img_in, threshold, maxlines, arrayy, arrayx, rows, cols);
  pthread_mutex_unlock(&__xlnx_cl_houghlines_accel_mutex);
}
void canny_accel(size_t img_inp, size_t img_out, unsigned int rows, unsigned int cols, unsigned int low_threshold, unsigned int high_threshold);

static pthread_mutex_t __xlnx_cl_canny_accel_mutex = PTHREAD_MUTEX_INITIALIZER;
void __stub____xlnx_cl_canny_accel(char **argv) {
  void **args = (void **)argv;
  size_t img_inp = *((size_t*)args[0+1]);
  size_t img_out = *((size_t*)args[1+1]);
  unsigned int rows = *((unsigned int*)args[2+1]);
  unsigned int cols = *((unsigned int*)args[3+1]);
  unsigned int low_threshold = *((unsigned int*)args[4+1]);
  unsigned int high_threshold = *((unsigned int*)args[5+1]);
 pthread_mutex_lock(&__xlnx_cl_canny_accel_mutex);
  canny_accel(img_inp, img_out, rows, cols, low_threshold, high_threshold);
  pthread_mutex_unlock(&__xlnx_cl_canny_accel_mutex);
}
void edgetracing_accel(size_t img_inp, size_t img_out, unsigned int rows, unsigned int cols);

static pthread_mutex_t __xlnx_cl_edgetracing_accel_mutex = PTHREAD_MUTEX_INITIALIZER;
void __stub____xlnx_cl_edgetracing_accel(char **argv) {
  void **args = (void **)argv;
  size_t img_inp = *((size_t*)args[0+1]);
  size_t img_out = *((size_t*)args[1+1]);
  unsigned int rows = *((unsigned int*)args[2+1]);
  unsigned int cols = *((unsigned int*)args[3+1]);
 pthread_mutex_lock(&__xlnx_cl_edgetracing_accel_mutex);
  edgetracing_accel(img_inp, img_out, rows, cols);
  pthread_mutex_unlock(&__xlnx_cl_edgetracing_accel_mutex);
}
}
