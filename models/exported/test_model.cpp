// test_model.cpp - validates wifi_rssi_model.h against Python predictions.
#include "wifi_rssi_model.h"
#include <cstdio>
#include <vector>
#include <string>
int main() {
  FILE* in = fopen("test_inputs.csv", "r");
  if (!in) { fprintf(stderr, "cannot open test_inputs.csv\n"); return 1; }
  char line[512];
  fgets(line, sizeof(line), in); // header
  FILE* out = fopen("cpp_predictions.csv", "w");
  fprintf(out, "x,y,floor_id,building_num,ap_mean_rssi,python_pred,cpp_pred,abs_diff\n");
  double x,y,apm,pp; int fl,bn; int n=0; double maxdiff=0.0, sumdiff=0.0;
  while (fgets(line, sizeof(line), in)) {
    if (sscanf(line, "%lf,%lf,%d,%d,%lf,%lf", &x,&y,&fl,&bn,&apm,&pp) == 6) {
      double cp = wifi_model::predictRSSI(x,y,fl,bn,apm);
      double d = cp - pp; if (d<0) d=-d;
      if (d>maxdiff) maxdiff=d; sumdiff+=d; n++;
      fprintf(out, "%.10g,%.10g,%d,%d,%.10g,%.12g,%.12g,%.3e\n", x,y,fl,bn,apm,pp,cp,d);
    }
  }
  fclose(in); fclose(out);
  printf("rows=%d max_abs_diff=%.9f mean_abs_diff=%.9f\n", n, maxdiff, sumdiff/(n?n:1));
  return 0;
}
