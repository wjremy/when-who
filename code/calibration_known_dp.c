#include <math.h>
#include <stdlib.h>
// Exact elementary symmetric polynomial and its first two beta derivatives.
void group_eval(int G, const int *starts, const int *counts, const double *x, const double *ysum, double beta, double *out) {
 for(int g=0;g<G;g++){
  int n=starts[g+1]-starts[g], m=counts[g];
  if(m==0){out[3*g]=out[3*g+1]=out[3*g+2]=0;continue;}
  double *d=calloc(m+1,sizeof(double)),*v=calloc(m+1,sizeof(double)),*q=calloc(m+1,sizeof(double)); d[0]=1; double scale=0;
  for(int t=0;t<n;t++){
   double z=x[starts[g]+t], w=exp(beta*z); int hi=t+1<m?t+1:m;
   for(int k=hi;k>=1;k--){q[k]+=w*(q[k-1]+2*z*v[k-1]+z*z*d[k-1]);v[k]+=w*(v[k-1]+z*d[k-1]);d[k]+=w*d[k-1];}
   double mx=0;for(int k=0;k<=hi;k++)if(d[k]>mx)mx=d[k];
   for(int k=0;k<=hi;k++){d[k]/=mx;v[k]/=mx;q[k]/=mx;}scale+=log(mx);
  }
  double mu=v[m]/d[m];out[3*g]=beta*ysum[g]-log(d[m])-scale;out[3*g+1]=ysum[g]-mu;out[3*g+2]=fmax(q[m]/d[m]-mu*mu,0);
  free(d);free(v);free(q);
 }
}
