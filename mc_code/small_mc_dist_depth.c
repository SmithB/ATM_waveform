char   t1[80] = "Small Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char   t2[80] = "1 W/cm^2 Uniform Illumination of Semi-Infinite Medium";

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define BINS 101

double g = 0.0;		      	/* Scattering Anisotropy -1<=g<=1 */
double z0=0; /* start photons at this level */
double thick=0;
long   i, photons = 10000000;
double z_max = 0.0;
double x,y,z,u,v,w,weight,dist,g2;
double deepest_scatter=0;
int ASCII=1; 

double rs, rd;
long count;
long maxrand;
double d;
long N_max=4000;

char *outfile;
FILE *outPtr;
double outbuf[10];

void dump() {
  if (ASCII==1) {
printf("%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %10.6f %10.6f %10ld %16ld \n", x, y, z, u, v, w, dist, deepest_scatter, count, i);
  } else {
  outbuf[0]=x;
  outbuf[1]=y;
  outbuf[2]=z;
  outbuf[3]=u;
  outbuf[4]=v;
  outbuf[5]=w;
  outbuf[6]=dist;
  outbuf[7]=deepest_scatter;
  outbuf[8]=(double) i;
  outbuf[9]=(double) count ;
  fwrite(outbuf, sizeof(double), 10, outPtr);
  }
}

void launch() /* Start the photon */
{
  x = 0.0; y = 0.0; z = z0;  
  u = 0.0; v = 0.0; w = 1.0;
  count=0;
  dist=z0;
  deepest_scatter=0;
}

void move() /* move to next scattering or absorption event */
{
  d = -log((random()+1.0)/(maxrand+1.0)); 
  x += d * u;
  y += d * v;
  z += d * w;
  if (z > deepest_scatter){
    deepest_scatter=z;
  }
  dist +=d;
  if ( z<=0 ) {
    dist -= z/w;
    x -= z/w*u;
    y -= z/w*v;
    z -= z/w*w;
    dump();
    /* exit location is x-z/w*u y-z/w*v 0*/
    /*printf("%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %10.6f %10.6f %10ld %16ld \n", x-z/w*u, y-z/w*v, z-z/w*w, u, v, w, dist-z/w, deepest_scatter, count, i);*/
    count=2e9+1;
  } 
  if ( z>z_max ) {
    dist -= dist-(z-z_max)/w;
    x -= (z-z_max)/w*u;
    y -= (z-z_max)/w*v;
    z -= (z-z_max)/w*w;
    deepest_scatter=z_max;
    dump();
    /*printf(" %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %10.6f %10.6f %10ld %16ld \n", x-(z-z_max)/w*u, y-(z-z_max)/w*v, z-(z-z_max)/w*w, u, v, w, dist-(z-z_max)/w, deepest_scatter, count, i);*/
    count=2e9+1;
  }
}

void absorb () /* Absorb light in the medium */
{
  count++;

  /* if (weight < 0.001){ 
    bit -= weight; 
    if (rand() > 0.1*RAND_MAX) weight = 0; else weight /= 0.1;
    bit += weight;
  }*/ 
}
void scatter() /* Scatter photon and establish new direction */
{ 
  double x1, x2, x3, t, mu, mu2, g2, temp1; 

  for(;;) {								/*new direction*/
    x1=2.0*random()/maxrand - 1.0; 
    x2=2.0*random()/maxrand - 1.0; 
    if ((x3=x1*x1+x2*x2)<1.0) break;
  }	
  if (g==0) {  /* isotropic */
    u = 2.0 * x3 -1.0;
    v = x1 * sqrt((1.0-u*u)/x3);
    w = x2 * sqrt((1.0-u*u)/x3);
    return;
  } 
  mu = (1.0-g2)/(1.0-g+2.0*g*random()/maxrand);
  mu = (1.0 + g2-mu*mu)/2.0/g; 
  mu2 = mu*mu;
  if ( fabs(w) < 0.9 ) {	/* avoid round-off errors in w */
    temp1=sqrt((1-mu2)/(1-w*w)/x3);
    t = mu * u + temp1 * (x1*u*w-x2*v);
    v = mu * v + temp1 * (x1*v*w+x2*u);
    w = mu * w - sqrt((1-mu2)*(1-w*w)/x3) * x1;
  } else {
    temp1=sqrt((1-mu2)/(1-v*v)/x3);
    t = mu * u + temp1 * (x1*u*v + x2*w);
    w = mu * w + temp1 * (x1*v*w - x2*u);
    v = mu * v - sqrt((1-mu2)*(1-v*v)/x3) * x1;
  }
  u = t;
  return;
}

int main (int argc, char *argv[] )
{
  int index;
  char c; 
  int  last_arg_processed=0;
  opterr = 0;
  thick=0.;
  if (argc==0) {
    fprintf (stderr, "small_mc_dist_depth -g asymmetry_parameter -p n_Photons -m max_scattering -o outfile -l scaled_layer_thickness \n");
  }
  while ((c = getopt (argc, argv, "hg:p:z:l:o:")) != -1) {
    last_arg_processed+=2;
    switch (c) { 
    case 'g':
      g = (double) atof(optarg);
      break;
    case 'z':
      z0 = (double) atof(optarg);
      fprintf(stderr, "USING z0\n");
      break;
    case 'p':
      photons = (long) atol(optarg);
      break;
    case 'l':
      thick = (double) atof(optarg);
      break;
    case 'o':
      outfile = optarg;
      break;
    case 'h':
      fprintf(stderr,"small_mc_dist_depth -g asymmetry_parameter -p n_Photons -z z0 -o output file -l scaled_layer_thickness \n");
      return 1;
    default:
      abort();
    }
  }
  if (outfile[0]=='\0') {
     ASCII=1;
   } else {
     ASCII=0;
     printf("outfile is %s\n", outfile);
     outPtr=fopen(outfile,"wb");
  }
   
  g2=g*g;
  srandom(time(0));
  maxrand=(exp(log(2.)*31.))-1;
  
  double g_0=g;
  double g2_0=g2;

  if (thick==0) {
    z_max = 500.0/(1.0-g);
  } else {
    z_max = thick;
  }

  fprintf(stderr, "z0=%3.2f, z_max=%3.2f, g=%3.2f, photons=%10ld\n", z0, z_max, g, photons);
  for (i = 1; i <= photons; i++){
    launch ();
    if (z0>0) {
      g=0.0;
      g2=0.0;
      scatter(); /* the first event is an isotropic scattering at z0 */
      g=g_0;
      g2=g2_0;
      count++;
    }
    while (count < N_max ) { 
      move ();
      count++;
      scatter ();
    }
  }	
  return 0;
}
