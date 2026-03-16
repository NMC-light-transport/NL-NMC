/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of NMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      kernels.metal, contains Metal Banchmarks for Monte Carlo simulation of fluence rate F and escaping flux J,
//                  in a semi-infinite medium such as biological tissue, with an external_medium/tissue surface boundary from
//                  https://omlc.org/software/mc/mcsub/ and mcxyz https://omlc.org/software/mc/mcxyz/index.html
/*---------------------------------------------------------------------------------------------------------------------*/




#include <metal_stdlib>
#import <metal_atomic>
#include "definitions.h"
#import "helpers.metal"


using namespace metal;

/* Propagation parameters */
//typedef struct tagPhoton
//{
//public:
//    thread tagPhoton()
//    {
//        x = MC_ZERO; y = MC_ZERO; z = MC_ZERO;
//        ux = MC_ZERO; uy = MC_ZERO; uz = MC_ZERO;
//        uxx = MC_ZERO; uyy = MC_ZERO; uzz = MC_ZERO;
//        s = MC_ZERO; sleft = MC_ZERO; costheta = MC_ZERO;
//        sintheta = MC_ZERO; cospsi = MC_ZERO; sinpsi = MC_ZERO;
//        psi = MC_ZERO; num_scatt = 0; W = MC_ONE; absorb = MC_ZERO;
//        photon_status = ALIVE; sv = false; tiss_type = -1;
//    };
//    float   x, y, z;        /* photon position */
//    float   ux, uy, uz;     /* photon trajectory as cosines */
//    float   uxx, uyy, uzz;  /* temporary values used during SPIN */
//    float   s;              /* step sizes. s = -log(RND)/mus [cm] */
//    float   sleft;          /* dimensionless */
//    float   costheta;       /* cos(theta) */
//    float   sintheta;       /* sin(theta) */
//    float   cospsi;         /* cos(psi) */
//    float   sinpsi;         /* sin(psi) */
//    float   psi;            /* azimuthal angle */
//    float   W;              /* photon weight */
//    float   absorb;         /* weighted deposited in a step due to absorption */
//    bool    photon_status;  /* flag = ALIVE=1 or DEAD=0 */
//    bool    sv;             /* Are they in the same voxel? */
//    int     num_scatt;      /* current number of scatttering even */
//    int     tiss_type;      /* current tissue type */
//    int     type;           /* 0 - absorbed photon, 1/3 - laser photon, 2/4 - spontaneous Raman photon, 3/6 - stimulate Raman photon */
//} Photon;


kernel void MCSubKernel(
                       device RunParams* run_params,
                       device TissueParams* tissParams,
                       device float* J,
                       device float* F,
                       device float* SAE,
                       device int* iRunNum,
                       uint index [[thread_position_in_grid]])
{
    
    float3 pos = {0};
    float3 dir  = {0};
    
    float mua_mcsub = tissParams->muav[FIRST_TISSUE_LAYER_MCSUB];
    float mus_mcsub = tissParams->musv[FIRST_TISSUE_LAYER_MCSUB];
    float g_mcsub = tissParams->gv[FIRST_TISSUE_LAYER_MCSUB];
    float n1_mcsub = run_params->n1;
    float n2_mcsub = run_params->n2;

    /* Variable parameters */
    float    mut, albedo, absorb, rsp, Rsptot, Atot;
    float    rnd, xfocus, S = 0.0, A = 0.0, E = 0.0, W = 0.0;
    float    uz1, uxx,uyy,uzz, s,r,temp;
    float    psi,costheta,sintheta,cospsi,sinpsi;
    long     ir, iz, CNT;
    short    photon_status;
    int numRuns = *iRunNum+1;
    int nums_scatt = 0;
    
   // RandomGen rng_gen = RandomGen(numRuns*index, numRuns*1234, index*1234);
    
    
    RandomGen rng_gen = RandomGen(
        (numRuns ^ (index * 2654435761U)) + 12345U,  // Mixed with golden ratio prime
        (numRuns * 1103515245U) ^ (index * 1013904223U),  // LCG scrambling
        (index ^ numRuns) * 1664525U + 1013904223U // XOR-mix + LCG-style shift
    );
    
    bool photon_detected = false;
    
    CNT = 0;
    mut    = mua_mcsub + mus_mcsub;
    albedo = mus_mcsub/mut;
    Rsptot = 0.0; /* accumulate specular reflectance per photon */
    Atot   = 0.0; /* accumulate absorbed photon weight */
    rsp = 0.0;
    
    if (mcflag_mcsub == 0) {
        /* UNIFORM COLLIMATED BEAM INCIDENT AT z = zs */
        /* Launch at (r,zz) = (radius*sqrt(rnd), 0).
         * Due to cylindrical symmetry, radial launch position is
         * assigned to x while y = 0.
         * radius = radius of uniform beam. */
        /* Initial position */
        rnd = rng_gen.rand();
        pos.xyz = float3(radius_mcsub*sqrt(rnd), 0.0, zs_mcsub);
        /* Initial trajectory as cosines */
        dir.xyz = float3(0.0, 0.0, 1.0);
        /* specular reflectance */
        temp   = n1_mcsub/n2_mcsub; /* refractive index mismatch, internal/external */
        temp   = (1.0 - temp)/(1.0 + temp);
        rsp    = temp*temp; /* specular reflectance at boundary */
        }
    else if (mcflag_mcsub == 1) {
        /* GAUSSIAN BEAM AT SURFACE */
        /* Launch at (r,z) = (radius*sqrt(-log(rnd)), 0).
         * Due to cylindrical symmetry, radial launch position is
         * assigned to x while y = 0.
         * radius = 1/e radius of Gaussian beam at surface.
         * waist  = 1/e radius of Gaussian focus.
         * zfocus = depth of focal point. */
        /* Initial position */
        while ((rnd = rng_gen.rand()) <= 0.0); /* avoids rnd = 0 */
        pos.xyz = float3(radius_mcsub*sqrt(-log(rnd)), 0.0, zs_mcsub);
        /* Initial trajectory as cosines */
        /* Due to cylindrical symmetry, radial launch trajectory is
         * assigned to ux and uz while uy = 0. */
        while ((rnd = rng_gen.rand()) <= 0.0); /* avoids rnd = 0 */
        xfocus = waist_mcsub*sqrt(-log(rnd));
        temp = sqrt((pos.x - xfocus)*(pos.x - xfocus) + zfocus_mcsub*zfocus_mcsub);
        sintheta = -(pos.x - xfocus)/temp;
        costheta = zfocus_mcsub/temp;
        dir.xyz = float3(sintheta, 0.0, costheta);
        /* specular reflectance and refraction */
        float uz = dir.z;
        rsp = Fresnel::RFresnel(n2_mcsub, n1_mcsub, costheta, &uz); /* new uz */
        dir.z = uz;
        dir.x  = sqrt(1.0 -  dir.z*dir.z); /* new ux */
        }
    else if  (mcflag_mcsub == 2) {
        /* ISOTROPIC POINT SOURCE AT POSITION xs,ys,zs */
        /* Initial position */
        pos.xyz = float3(xs_mcsub, ys_mcsub, zs_mcsub);
        /* Initial trajectory as cosines */
        costheta = 1.0 - 2.0*rng_gen.rand();
        sintheta = sqrt(1.0 - costheta*costheta);
        psi = 2.0*PI*rng_gen.rand();
        cospsi = cos(psi);
        if (psi < PI)
            sinpsi = sqrt(1.0 - cospsi*cospsi);
        else
            sinpsi = -sqrt(1.0 - cospsi*cospsi);
        dir.xyz = float3(sintheta*cospsi, sintheta*sinpsi, costheta);
        /* specular reflectance */
        rsp = 0.0;
        }
    
    W             = 1.0 - rsp;  /* set photon initial weight */
    Rsptot       += rsp; /* accumulate specular reflectance per photon */
    photon_status = ALIVE;
    
    /*
    ******************************************
    ****** HOP_ESCAPE_SPINCYCLE **************
    * Propagate one photon until it dies by ESCAPE or ROULETTE.
    *******************************************/
    
    do {

    /**** HOP
     * Take step to new position
     * s = stepsize
     * ux, uy, uz are cosines of current photon trajectory
     *****/
        while ((rnd = rng_gen.rand()) <= 0.0);   /* avoids rnd = 0 */
        s = -log(rnd)/mut;   /* Step size.  Note: log() is base e */
        pos.x += s*dir.x;           /* Update positions. */
        pos.y += s*dir.y;
        pos.z += s*dir.z;

        /* Does photon ESCAPE at surface? ... z <= 0? */
         if ( (boundaryflag_mcsub == 1) && (pos.z <= zs_mcsub)) {
            rnd = rng_gen.rand();
            /* Check Fresnel reflectance at surface boundary */
             if (rnd > Fresnel::RFresnel(n1_mcsub, n2_mcsub, -dir.z, &uz1)) {
                /* Photon escapes at external angle, uz1 = cos(angle) */
                pos.x -= s*dir.x;       /* return to original position */
                pos.y -= s*dir.y;
                pos.z -= s*dir.z;
                s  = fabs(pos.z/dir.z); /* calculate stepsize to reach surface */
                pos.x += s*dir.x;       /* partial step to reach surface */
                pos.y += s*dir.y;
                r = sqrt(pos.x*pos.x + pos.y*pos.y);   /* find radial position r */
                ir = (long)(r/dr_mcsub) + 1; /* round to 1 <= ir */
                if (ir > BINS) ir = BINS;  /* ir = NR is overflow bin */
                long thread_offset_J = index*BINS;
                J[thread_offset_J + ir] += W;      /* increment escaping flux */
                E += W;
                photon_detected = true;
                photon_status = DEAD;
                }
            else {
                pos.z = -pos.z;   /* Total internal reflection. */
                dir.z = -dir.z;
                }
            }

    if (photon_status == ALIVE) {
        /*********************************************
         ****** SPINCYCLE = DROP_SPIN_ROULETTE ******
         *********************************************/

        /**** DROP
         * Drop photon weight (W) into local bin.
         *****/
        absorb = W*(1.0 - albedo);      /* photon weight absorbed at this step */
        W -= absorb;                  /* decrement WEIGHT by amount absorbed */
        Atot += absorb;               /* accumulate absorbed photon weight */
        /* deposit power in cylindrical coordinates z,r */
        r  = sqrt(pos.x*pos.x + pos.y*pos.y);         /* current cylindrical radial position */
        ir = (long)(r/dr_mcsub) + 1;        /* round to 1 <= ir */
        iz = (long)(fabs(pos.z)/dz_mcsub) + 1;  /* round to 1 <= iz */
        if (ir >= BINS) ir = BINS;        /* last bin is for overflow */
        if (iz >= BINS) iz = BINS;        /* last bin is for overflow */
        long thread_offset = index*BINS*BINS;
        F[thread_offset + ir*BINS + iz] += absorb;          /* DROP absorbed weight into bin */
    
        /**** SPIN
         * Scatter photon into new trajectory defined by theta and psi.
         * Theta is specified by cos(theta), which is determined
         * based on the Henyey-Greenstein scattering function.
         * Convert theta and psi into cosines ux, uy, uz.
         *****/
        /* Sample for costheta */
        rnd = rng_gen.rand();
        if (FltEq(g_mcsub,0.0))
            costheta = 2.0*rnd - 1.0;
        else if (FltEq(g_mcsub,1.0))
            costheta = 1.0;
        else {
            temp = (1.0 - g_mcsub*g_mcsub)/(1.0 - g_mcsub + 2.0*g_mcsub*rnd);
            costheta = (1.0 + g_mcsub*g_mcsub - temp*temp)/(2.0*g_mcsub);
            }
        costheta = 2.0*rnd - 1.0;
        sintheta = sqrt(1.0 - costheta*costheta);/*sqrt faster than sin()*/

        /* Sample psi. */
        psi = 2.0*PI*rng_gen.rand();
        cospsi = cos(psi);
        if (psi < PI)
            sinpsi = sqrt(1.0 - cospsi*cospsi); /*sqrt faster */
        else
            sinpsi = -sqrt(1.0 - cospsi*cospsi);

        /* New trajectory. */
        if (1.0 - fabs(dir.z) <= 1.0e-12) {  /* close to perpendicular. */
            uxx = sintheta*cospsi;
            uyy = sintheta*sinpsi;
            uzz = costheta*((dir.z)>=0 ? 1:-1);
            }
        else {   /* usually use this option */
            temp = sqrt(1.0 - dir.z*dir.z);
            uxx = sintheta*(dir.x*dir.z*cospsi - dir.y*sinpsi)/temp + dir.x*costheta;
            uyy = sintheta*(dir.y*dir.z*cospsi + dir.x*sinpsi)/temp + dir.y*costheta;
            uzz = -sintheta*cospsi*temp +  dir.z*costheta;
            }

        /* Update trajectory */
        dir.x = uxx;
        dir.y = uyy;
        dir.z = uzz;
        
        nums_scatt++;

        /**** CHECK ROULETTE
         * If photon weight below THRESHOLD, then terminate photon using
         * Roulette technique. Photon has CHANCE probability of having
         * its weight increased by factor of 1/CHANCE,
         * and 1-CHANCE probability of terminating.
         *****/
        if (W < THRESHOLD) {
            rnd = rng_gen.rand();
            if (rnd <= CHANCE)
                W /= CHANCE;
            else photon_status = DEAD;
            }
        
     //   if (nums_scatt > MAX_SCATT)
      //      photon_status = DEAD;

        }
        /**********************************************
          **** END of SPINCYCLE = DROP_SPIN_ROULETTE *
          **********************************************/

    }
    while (photon_status == ALIVE);
    
    long thread_offset_J = index*3;
    S = Rsptot;
    A = Atot;
    
   /* atomic_fetch_add_explicit(&SAE[thread_offset_J + 0], S, memory_order_relaxed);
    atomic_fetch_add_explicit(&SAE[thread_offset_J + 1], A, memory_order_relaxed);
    atomic_fetch_add_explicit(&SAE[thread_offset_J + 2], E, memory_order_relaxed);*/
    
    SAE[thread_offset_J + 0] += S;
    SAE[thread_offset_J + 1] += A;
    SAE[thread_offset_J + 2] += E;
    
}

kernel void MCXYZKernel(
                       device RunParams* run_params,
                       device TissueParams* tissParams,
                       device char* V,
                       device atomic_float* F3D,
                       device atomic_float* Rd,
                       device atomic_float* Rd2D,
                       device atomic_int* DetPhot2D,
                       device int* iRunNum,
                       device float* timeSeed,
                       uint index [[thread_position_in_grid]])
{

    int numRuns = *iRunNum+1;
    Photon photon;
    RunParams runParamsG = *run_params;
    TissueParams tissParamsG = *tissParams;

    RandomGen rng_gen = RandomGen(
        (numRuns ^ (index * 2654435761U)) + 12345U,  // Mixed with golden ratio prime
        (numRuns * 1103515245U) ^ (index * 1013904223U),  // LCG scrambling
        (index ^ numRuns) * 1664525U + 1013904223U // XOR-mix + LCG-style shift
    );
    
    /**** LAUNCH
    Initialize photon position and trajectory.
    *****/

    photon.num_scatt = 0;
    photon.W = 1.0;                    /* set photon weight to one */
    photon.photon_status = ALIVE;      /* Launch an ALIVE photon */
    photon.tiss_type = -1;
    float rnd = 0.0;
    float Rd_loc = 0.0;
    float TotalPath = 0.0;
    
    
    /**** SET SOURCE
     * Launch collimated beam at x,y center.
     ****/
    
    /****************************/
    /* Initial position. */
    
    /* trajectory */
    if (runParamsG.launchflag==1) { // manually set launch
        photon.x    = runParamsG.xs;
        photon.y    = runParamsG.ys;
        photon.z    = runParamsG.zs;
        photon.ux   = runParamsG.ux0;
        photon.uy   = runParamsG.uy0;
        photon.uz   = runParamsG.uz0;
    }
    else { // use mcflag
        if (runParamsG.mcflag==0) { // uniform beam
            // set launch point and width of beam
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float r        = runParamsG.radius*sqrt(rnd); // radius of beam at launch point
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float phi       = rnd*2.0*PI;
            photon.x        = runParamsG.xs + r*cos(phi);
            photon.y        = runParamsG.ys + r*sin(phi);
            photon.z        = runParamsG.zs;
            // set trajectory toward focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            r        = runParamsG.waist*sqrt(rnd); // radius of beam at focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            phi        = rnd*2.0*PI;
            float xfocus    = r*cos(phi);
            float yfocus    = r*sin(phi);
            float temp    = sqrt((photon.x - xfocus)*(photon.x - xfocus) + (photon.y - yfocus)*(photon.y - yfocus) + runParamsG.zfocus*runParamsG.zfocus);
            photon.ux        = -(photon.x - xfocus)/temp;
            photon.uy        = -(photon.y - yfocus)/temp;
            photon.uz        = sqrt(1.0 - photon.ux*photon.ux - photon.uy*photon.uy);
        }
        else if (runParamsG.mcflag==1) {
            /* GAUSSIAN BEAM AT SURFACE */
            /* Launch at (r,z) = (radius*sqrt(-log(rnd)), 0).
             * Due to cylindrical symmetry, radial launch position is
             * assigned to x while y = 0.
             * radius = 1/e radius of Gaussian beam at surface.
             * waist  = 1/e radius of Gaussian focus.
             * zfocus = depth of focal point. */
            /* Initial position */
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float r        = runParamsG.radius*sqrt(-log(rnd)); // radius of beam at launch point
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float phi       = rnd*2.0*PI;
            photon.x        = runParamsG.xs + r*cos(phi);
            photon.y        = runParamsG.ys + r*sin(phi);
            photon.z        = runParamsG.zs;
            // set trajectory toward focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            r        = runParamsG.waist*sqrt(rnd); // radius of beam at focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            phi        = rnd*2.0*PI;
            float xfocus    = r*cos(phi);
            float yfocus    = r*sin(phi);
            float temp    = sqrt((photon.x - xfocus)*(photon.x - xfocus) + (photon.y - yfocus)*(photon.y - yfocus) + runParamsG.zfocus*runParamsG.zfocus);
            photon.ux        = -(photon.x - xfocus)/temp;
            photon.uy        = -(photon.y - yfocus)/temp;
            photon.uz        = sqrt(1.0 - photon.ux*photon.ux - photon.uy*photon.uy);
            /* specular reflectance and refraction */
            float uz_new = photon.uz;
            float int_rsp = Fresnel::RFresnel(run_params->n2, run_params->n1, fabs(photon.uz), &uz_new); /* new uz */
            photon.W = photon.W * (MC_ONE - int_rsp);
            Fresnel::Snell(run_params->n2, run_params->n1, &photon.ux, &photon.uy, &photon.uz);
            }
        else if (runParamsG.mcflag==2) { // isotropic pt source
            photon.costheta = 1.0 - 2.0*RandomNum;
            photon.sintheta = sqrt(1.0 - photon.costheta*photon.costheta);
            float psi = 2.0*PI*RandomNum;
            photon.cospsi = cos(psi);
            if (psi < PI)
                photon.sinpsi = sqrt(1.0 - photon.cospsi*photon.cospsi);
            else
                photon.sinpsi = -sqrt(1.0 - photon.cospsi*photon.cospsi);
            photon.x = runParamsG.xs;
            photon.y = runParamsG.ys;
            photon.z = runParamsG.zs;
            photon.ux = photon.sintheta*photon.cospsi;
            photon.uy = photon.sintheta*photon.sinpsi;
            photon.uz = photon.costheta;
        }
        else if (runParamsG.mcflag==3) { // rectangular source collimated
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.x = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify x-halfwidth of rectangle
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.y = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify y-halfwidth of rectangle
            photon.z = runParamsG.zs;
            photon.ux = 0.0;
            photon.uy = 0.0;
            photon.uz = 1.0; // collimated beam
        }
        else if (runParamsG.mcflag==4) { // point source collimated
            photon.x = 0.0;
            photon.y = 0.0;
            photon.z = runParamsG.zs;
            photon.ux = 0.0;
            photon.uy = 0.0;
            photon.uz = 1.0; // collimated beam
        }
    } // end  use mcflag
    /****************************/
    
    /* Get tissue voxel properties of launchpoint.
        * If photon beyond outer edge of defined voxels,
        * the tissue equals properties of outermost voxels.
        * Therefore, set outermost voxels to infinite background value.
        */
    /* Added. Used to track photons */
    int ix = (int)(runParamsG.Nx / 2 + photon.x / runParamsG.dx);
    int iy = (int)(runParamsG.Ny / 2 + photon.y / runParamsG.dy);
    int iz = (int)(photon.z / runParamsG.dz);
    if (ix >= runParamsG.Nx) ix = runParamsG.Nx - 1;
    if (iy >= runParamsG.Ny) iy = runParamsG.Ny - 1;
    if (iz >= runParamsG.Nz) iz = runParamsG.Nz - 1;
    if (ix<0)   ix = 0;
    if (iy<0)   iy = 0;
    if (iz<0)   iz = 0;
    /* Get the tissue type of located voxel */
    long i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
    photon.tiss_type = V[i];
    float mua = tissParamsG.muav[photon.tiss_type];
    float mus = tissParamsG.musv[photon.tiss_type];
    float g = tissParamsG.gv[photon.tiss_type];
    int bflag = 1; // initialize as 1 = inside volume, but later check as photon propagates.
    
    /* HOP_DROP_SPIN_CHECK
     Propagate one photon until it dies as determined by ROULETTE.
     *******/
    
    do {
        
      /**** HOP
         Take step to new position
         s = dimensionless stepsize
         x, uy, uz are cosines of current photon trajectory
         *****/
        
      while ((rnd = RandomNum) <= 0.0);   /* yields 0 < rnd <= 1 */
      photon.sleft = -log(rnd);                /* dimensionless step */
      int curr_depth = 0;
            
      do {  // while sleft>0 or maximum tracing depth achived
            photon.s = photon.sleft / mus;                /* Step size [cm].*/
            float    tempx, tempy, tempz; /* temporary variables, used during photon step. */
            tempx = photon.x + photon.s*photon.ux;                /* Update positions. [cm] */
            tempy = photon.y + photon.s*photon.uy;
            tempz = photon.z + photon.s*photon.uz;
            TotalPath += photon.s;
            
            photon.sv = SameVoxel(photon.x, photon.y, photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz);
            
            if (photon.sv) /* photon in same voxel */
            {
                photon.x = tempx;                    /* Update positions. */
                photon.y = tempy;
                photon.z = tempz;

                /**** DROP
                Drop photon weight (W) into local bin.
                *****/
                photon.absorb = photon.W*(1.0 - exp(-mua*photon.s));    /* photon weight absorbed at this step */
                photon.W -= photon.absorb;                    /* decrement WEIGHT by amount absorbed */
                // If photon within volume of heterogeneity, deposit energy in F[].
                // Normalize F[] later, when save output.
                if (bflag)
                   // F3D[i] += photon.absorb;  // only save data if blag==1, i.e., photon inside simulation cube
                atomic_fetch_add_explicit(&F3D[i], photon.absorb, memory_order_relaxed);
                /* Update sleft */
                photon.sleft = MC_ZERO;        /* dimensionless step remaining */
            }
            else /* photon has crossed voxel boundary */
            {
                
                /* step to voxel face + "littlest step" so just inside new voxel. */
                photon.s = ls + FindVoxelFace2(photon.x, photon.y, photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz, photon.ux, photon.uy, photon.uz);
                //photon.s = ls + FindVoxelFace3(photon.x, photon.y, photon.z, runParamsG.dx, runParamsG.dy, runParamsG.dz, photon.ux, photon.uy, photon.uz);

                /**** DROP
                Drop photon weight (W) into local bin.
                *****/
                photon.absorb = photon.W*(1.0 - exp(-mua*photon.s));   /* photon weight absorbed at this step */
                photon.W -= photon.absorb;                  /* decrement WEIGHT by amount absorbed */
                // If photon within volume of heterogeneity, deposit energy in F[].
                // Normalize F[] later, when save output.
                if (bflag)
                    //F3D[i] += photon.absorb;  // only save data if blag==1, i.e., photon inside simulation cube
                    atomic_fetch_add_explicit(&F3D[i], photon.absorb, memory_order_relaxed);

                /* Update sleft */
                photon.sleft -= photon.s*mus;  /* dimensionless step remaining */
                if (photon.sleft <= ls) photon.sleft = 0.0;

                // check if photon crosses the boundary
               /* float z_new = photon.z + photon.s*photon.uz;
                if (z_new <= runParamsG.zsurf && photon.uz < MC_ZERO)*/
                // check if photon crosses the boundary
                float z_new = photon.z + photon.s*photon.uz;
                bool cross_boundary = false;
                if (runParamsG.det_state == SIM_PH_PACKETS_DETECTION_REF && z_new <= runParamsG.zsurf && photon.uz < MC_ZERO)
                    cross_boundary = true;
                else if (runParamsG.det_state == SIM_PH_PACKETS_DETECTION_TRANS && z_new >= runParamsG.zsurf && photon.uz > MC_ZERO)
                    cross_boundary = true;
                if (cross_boundary)
                {
                    float dl_b = (runParamsG.zsurf - photon.z) / photon.uz;
                    {
                        photon.sleft = (photon.s - dl_b) * mus;
                        if (photon.sleft <= ls) photon.sleft = MC_ZERO;
                        photon.s = dl_b;
                        photon.x += photon.s*photon.ux;
                        photon.y += photon.s*photon.uy;
                        photon.z += photon.s*photon.uz;
                        
                        float uz_temp = MC_ZERO;
                        // TODO: account for Snell law while escaping, if dux, duy, duz are needed
                        float rsp = Fresnel::RFresnel(run_params->n1, run_params->n2, fabs(photon.uz), &uz_temp);
                        if (rsp < MC_ONE)
                        {
                            float th_exit = MC_ZERO;
                            if (!FltEq(uz_temp, MC_ONE))
                                th_exit = (RAD_DEG*acos(uz_temp));
                           
                            float ro = sqrt(photon.x*photon.x + photon.y*photon.y);
                            float r_min = MC_ZERO;
                            float r_max = MC_ZERO;
                            // Source-Detector
                            if (FltEq(runParamsG.xd, MC_ZERO))
                            {
                                r_min = MC_ZERO;
                                r_max = runParamsG.det_radius;
                            }
                            else
                            {
                                r_min = runParamsG.xd - runParamsG.det_radius;
                                r_max = r_min + MC_TWO*runParamsG.det_radius;
                            }
                            
                            if ((ro >= r_min) && (ro <= r_max) && (th_exit <= runParamsG.na))
                            {
                                
                                ix = (int)(runParamsG.Nx / 2.0 + photon.x / runParamsG.dx);
                                iy = (int)(runParamsG.Ny / 2.0 + photon.y / runParamsG.dy);
                                if (ix >= runParamsG.Nx) ix = runParamsG.Nx - 1;
                                if (iy >= runParamsG.Ny) iy = runParamsG.Ny - 1;
                                if (ix<0)   ix = 0;
                                if (iy<0)   iy = 0;
                                
                                /* Record diffuse reflectance with RFresnel*/
                                Rd_loc += photon.W * (MC_ONE - rsp);
                                // Rd[index] += photon.W * (MC_ONE - rsp);
                                // atomic_fetch_add_explicit(&Rd[index], photon.W * (MC_ONE - rsp), memory_order_relaxed);
                                int i2d = (long)(index*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
                                //Rd2D[i2d] += photon.W * (MC_ONE - rsp);
                                atomic_fetch_add_explicit(&Rd2D[i2d], photon.W * (MC_ONE - rsp), memory_order_relaxed);
                                //DetPhot2D[i2d] +=1;
                                atomic_fetch_add_explicit(&DetPhot2D[i2d], 1, memory_order_relaxed);
                                photon.uz = -photon.uz;
                                photon.W = photon.W * rsp;
                            }
                        }
                        else // internally reflect
                        {
                            photon.uz = -photon.uz;
                            photon.W = photon.W * rsp;
                        }
                    }
                }
                else
                {
                    /* Update positions. */
                    photon.x += photon.s*photon.ux;
                    photon.y += photon.s*photon.uy;
                    photon.z += photon.s*photon.uz;
                }
                
                
                // pointers to voxel containing optical properties
                ix = (int)(runParamsG.Nx / 2.0 + photon.x / runParamsG.dx);
                iy = (int)(runParamsG.Ny / 2.0 + photon.y / runParamsG.dy);
                iz = (int)(photon.z / runParamsG.dz);


                if (runParamsG.boundaryflag == 0) { // Infinite medium.
                            // Check if photon has wandered outside volume.
                            // If so, set tissue type to boundary value, but let photon wander.
                            // Set blag to zero, so DROP does not deposit energy.
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; bflag = 0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; bflag = 0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; bflag = 0;}
                    if (iz<0)   {iz=0;    bflag = 0;}
                    if (ix<0)   {ix=0;    bflag = 0;}
                    if (iy<0)   {iy=0;    bflag = 0;}
                }
                else if (runParamsG.boundaryflag==1) { // Escape at boundaries
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iz<0)   {iz=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix<0)   {ix=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iy<0)   {iy=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                }
                else if (runParamsG.boundaryflag==2) { // Escape at top surface, no x,y bottom z boundaries
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; bflag = 0.0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; bflag = 0.0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; bflag = 0.0;}
                    if (iz<0)   {iz=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix<0)   {ix=0;    bflag = 0;}
                    if (iy<0)   {iy=0;    bflag = 0;}
                }

                // update pointer to tissue type
                i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
                photon.tiss_type = V[i];
                mua = tissParamsG.muav[photon.tiss_type];
                mus = tissParamsG.musv[photon.tiss_type];
                g = tissParamsG.gv[photon.tiss_type];


            } //(sv) /* same voxel */
          curr_depth++;
          //photon.num_scatt++;
       } while (curr_depth <= RAY_DEPTH && photon.sleft > MC_ZERO); //do...while

        
        /**** SPIN
        Scatter photon into new trajectory defined by theta and psi.
        Theta is specified by cos(theta), which is determined
        based on the Henyey-Greenstein scattering function.
        Convert theta and psi into cosines ux, uy, uz.
        *****/
        /* Sample for costheta */
        rnd = RandomNum;
        if (FltEq(g, 0.0))
            photon.costheta = 2.0*rnd - 1.0;
        else if (FltEq(g, 1.0))
            photon.costheta = 1.0;
        else {
            float temp = (1.0 - g*g) / (1.0 - g + 2.0 * g*rnd);
            photon.costheta = (1.0 + g*g - temp*temp) / (2.0*g);
        }
        photon.sintheta = sqrt(1.0 - photon.costheta*photon.costheta); /* sqrt() is faster than sin(). */

        /* Sample psi. */
        photon.psi = 2.0*PI*RandomNum;
        photon.cospsi = cos(photon.psi);
        if (photon.psi < PI)
            photon.sinpsi = sqrt(1.0 - photon.cospsi*photon.cospsi);     /* sqrt() is faster than sin(). */
        else
            photon.sinpsi = -sqrt(1.0 - photon.cospsi*photon.cospsi);

        /* New trajectory. */
        if (1.0 - fabs(photon.uz) <= ONE_MINUS_COSZERO) {      /* close to perpendicular. */
            photon.uxx = photon.sintheta * photon.cospsi;
            photon.uyy = photon.sintheta * photon.sinpsi;
            photon.uzz = photon.costheta * SIGN(photon.uz);   /* SIGN() is faster than division. */
        }
        else {              /* usually use this option */
            float temp = sqrt(1.0 - photon.uz * photon.uz);
            photon.uxx = photon.sintheta * (photon.ux * photon.uz * photon.cospsi - photon.uy * photon.sinpsi) / temp + photon.ux * photon.costheta;
            photon.uyy = photon.sintheta * (photon.uy * photon.uz * photon.cospsi + photon.ux * photon.sinpsi) / temp + photon.uy * photon.costheta;
            photon.uzz = -photon.sintheta * photon.cospsi * temp + photon.uz * photon.costheta;
        }

        /* Update trajectory */
        photon.ux = photon.uxx;
        photon.uy = photon.uyy;
        photon.uz = photon.uzz;
        photon.num_scatt++;
            
        /**** CHECK ROULETTE
        If photon weight below THRESHOLD, then terminate photon using Roulette technique.
        Photon has CHANCE probability of having its weight increased by factor of 1/CHANCE,
        and 1-CHANCE probability of terminating.
        *****/
        
        
        if (photon.W < THRESHOLD) {
            if (RandomNum <= CHANCE)
                photon.W /= CHANCE;
            else photon.photon_status = DEAD;
        }
        
        // Russian roulette with a semi-automatic
        if (photon.W < MIN_VALUE ||  photon.num_scatt > MAX_SCATT)
            photon.photon_status = DEAD;

        } while (photon.photon_status == ALIVE);  /* end STEP_CHECK_HOP_SPIN */
        /* if ALIVE, continue propagating */
        /* If photon DEAD, then launch new photon. */
        atomic_fetch_add_explicit(&Rd[index], Rd_loc, memory_order_relaxed);
}


kernel void MCXYZKernelTT(
                       device RunParams* run_params,
                       device TissueParams* tissParams,
                       device char* V,
                       device float* F3D,
                       device float* Rd,
                       device float* Rd2D,
                       device int* iRunNum,
                       uint index [[thread_position_in_grid]])
{

    int numRuns = *iRunNum+1;
    Photon photon;
    RunParams runParamsG = *run_params;
    TissueParams tissParamsG = *tissParams;
    RandomGen rng_gen = RandomGen(numRuns*index, numRuns*1234, index*1234);

    /**** LAUNCH
    Initialize photon position and trajectory.
    *****/

    photon.num_scatt = 0;
    photon.W = 1.0;                    /* set photon weight to one */
    photon.photon_status = ALIVE;      /* Launch an ALIVE photon */
    photon.tiss_type = -1;
    int bflag = 1; // initialize as 1 = inside volume, but later check as photon propagates.
    int surfflag = 1;

    float rnd = MC_ZERO;
    float q1f = MC_ZERO;
    float q2f = MC_ZERO;
    float q3f = MC_ZERO;
    float q1b = MC_ZERO;
    float q2b = MC_ZERO;
    float q3b = MC_ZERO;
    /****************************/
    /* Initial position. */
    
    /* trajectory */
    if (runParamsG.launchflag==1) { // manually set launch
        photon.x    = runParamsG.xs;
        photon.y    = runParamsG.ys;
        photon.z    = runParamsG.zs;
        photon.ux   = runParamsG.ux0;
        photon.uy   = runParamsG.uy0;
        photon.uz   = runParamsG.uz0;
    }
    else { // use mcflag
        if (runParamsG.mcflag==0) { // uniform beam
            // set launch point and width of beam
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float r        = runParamsG.radius*sqrt(rnd); // radius of beam at launch point
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float phi       = rnd*2.0*PI;
            photon.x        = runParamsG.xs + r*cos(phi);
            photon.y        = runParamsG.ys + r*sin(phi);
            photon.z        = runParamsG.zs;
            // set trajectory toward focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            r        = runParamsG.waist*sqrt(rnd); // radius of beam at focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            phi        = rnd*2.0*PI;
            float xfocus    = r*cos(phi);
            float yfocus    = r*sin(phi);
            float temp    = sqrt((photon.x - xfocus)*(photon.x - xfocus) + (photon.y - yfocus)*(photon.y - yfocus) + runParamsG.zfocus*runParamsG.zfocus);
            photon.ux        = -(photon.x - xfocus)/temp;
            photon.uy        = -(photon.y - yfocus)/temp;
            photon.uz        = sqrt(1.0 - photon.ux*photon.ux - photon.uy*photon.uy);
        }
        else if (runParamsG.mcflag==2) { // isotropic pt source
            photon.costheta = 1.0 - 2.0*RandomNum;
            photon.sintheta = sqrt(1.0 - photon.costheta*photon.costheta);
            float psi = 2.0*PI*RandomNum;
            photon.cospsi = cos(psi);
            if (psi < PI)
                photon.sinpsi = sqrt(1.0 - photon.cospsi*photon.cospsi);
            else
                photon.sinpsi = -sqrt(1.0 - photon.cospsi*photon.cospsi);
            photon.x = runParamsG.xs;
            photon.y = runParamsG.ys;
            photon.z = runParamsG.zs;
            photon.ux = photon.sintheta*photon.cospsi;
            photon.uy = photon.sintheta*photon.sinpsi;
            photon.uz = photon.costheta;
        }
        else if (runParamsG.mcflag==3) { // rectangular source collimated
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.x = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify x-halfwidth of rectangle
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.y = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify y-halfwidth of rectangle
            photon.z = runParamsG.zs;
            photon.ux = 0.0;
            photon.uy = 0.0;
            photon.uz = 1.0; // collimated beam
        }
        else if (runParamsG.mcflag==3) { // rectangular source collimated
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.x = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify x-halfwidth of rectangle
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.y = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify y-halfwidth of rectangle
            photon.z = runParamsG.zs;
            photon.ux = 0.0;
            photon.uy = 0.0;
            photon.uz = 1.0; // collimated beam
        }
        else if (runParamsG.mcflag==4) { // point source collimated
            photon.x = 0.0;
            photon.y = 0.0;
            photon.z = runParamsG.zs;
            photon.ux = 0.0;
            photon.uy = 0.0;
            photon.uz = 1.0; // collimated beam
        }
    } // end  use mcflag
    /****************************/

    /* Get tissue voxel properties of launchpoint.
        * If photon beyond outer edge of defined voxels,
        * the tissue equals properties of outermost voxels.
        * Therefore, set outermost voxels to infinite background value.
        */
    /* Added. Used to track photons */
    int ix = (int)(runParamsG.Nx / 2 + photon.x / runParamsG.dx);
    int iy = (int)(runParamsG.Ny / 2 + photon.y / runParamsG.dy);
    int iz = (int)(photon.z / runParamsG.dz);
    if (ix >= runParamsG.Nx) ix = runParamsG.Nx - 1;
    if (iy >= runParamsG.Ny) iy = runParamsG.Ny - 1;
    if (iz >= runParamsG.Nz) iz = runParamsG.Nz - 1;
    if (ix<0)   ix = 0;
    if (iy<0)   iy = 0;
    if (iz<0)   iz = 0;
    /* Get the tissue type of located voxel */
    long i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
    photon.tiss_type = V[i];
    float mua = tissParamsG.muav[photon.tiss_type];
    float mus = tissParamsG.musv[photon.tiss_type];
    
    /* TT parameters af bf ab gb --> q1f,q2f,q3f, q1b,q2b,q3b */
    float gf = tissParamsG.gf[photon.tiss_type];
    float af = tissParamsG.af[photon.tiss_type];
    float gb = tissParamsG.gb[photon.tiss_type];
    float ab = tissParamsG.ab[photon.tiss_type];
    
    q1f = (MC_ONE + gf * gf);
    q2f = pow(MC_ONE - gf, MC_TWO * af);
    q3f = pow(MC_ONE + gf, MC_TWO * af);
    q1b = (MC_ONE + gb * gb);
    q2b = pow(MC_ONE - gb, MC_TWO * ab);
    q3b = pow(MC_ONE + gb, MC_TWO * ab);
    
    /* HOP_DROP_SPIN_CHECK
     Propagate one photon until it dies as determined by ROULETTE.
     *******/
    
    do {
        
      /**** HOP
         Take step to new position
         s = dimensionless stepsize
         x, uy, uz are cosines of current photon trajectory
         *****/
        
      while ((rnd = RandomNum) <= 0.0);   /* yields 0 < rnd <= 1 */
      photon.sleft = -log(rnd);                /* dimensionless step */
      int curr_depth = 0;
            
      do {  // while sleft>0 or maximum tracing depth achived
            photon.s = photon.sleft / mus;                /* Step size [cm].*/
            float    tempx, tempy, tempz; /* temporary variables, used during photon step. */
            tempx = photon.x + photon.s*photon.ux;                /* Update positions. [cm] */
            tempy = photon.y + photon.s*photon.uy;
            tempz = photon.z + photon.s*photon.uz;
            
            photon.sv = SameVoxel(photon.x, photon.y, photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz);
            
            if (photon.sv) /* photon in same voxel */
            {
                photon.x = tempx;                    /* Update positions. */
                photon.y = tempy;
                photon.z = tempz;

                /**** DROP
                Drop photon weight (W) into local bin.
                *****/
                photon.absorb = photon.W*(1.0 - exp(-mua*photon.s));    /* photon weight absorbed at this step */
                photon.W -= photon.absorb;                    /* decrement WEIGHT by amount absorbed */
                // If photon within volume of heterogeneity, deposit energy in F[].
                // Normalize F[] later, when save output.
                if (bflag)
                    F3D[i] += photon.absorb;  // only save data if blag==1, i.e., photon inside simulation cube
                /* Update sleft */
                photon.sleft = 0.0;        /* dimensionless step remaining */
            }
            else /* photon has crossed voxel boundary */
            {
                
                /* step to voxel face + "littlest step" so just inside new voxel. */
                photon.s = ls + FindVoxelFace2(photon.x, photon.y, photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz, photon.ux, photon.uy, photon.uz);
                //photon.s = FindVoxelFace3(tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz, photon.ux, photon.uy, photon.uz);
                

                /**** DROP
                Drop photon weight (W) into local bin.
                *****/
                photon.absorb = photon.W*(1.0 - exp(-mua*photon.s));   /* photon weight absorbed at this step */
                photon.W -= photon.absorb;                  /* decrement WEIGHT by amount absorbed */
                // If photon within volume of heterogeneity, deposit energy in F[].
                // Normalize F[] later, when save output.
                if (bflag)
                    F3D[i] += photon.absorb;  // only save data if blag==1, i.e., photon inside simulation cube

                /* Update sleft */
                photon.sleft -= photon.s*mus;  /* dimensionless step remaining */
                if (photon.sleft <= ls) photon.sleft = 0.0;

                /* Update positions. */
                photon.x += photon.s*photon.ux;
                photon.y += photon.s*photon.uy;
                photon.z += photon.s*photon.uz;

                // pointers to voxel containing optical properties
                ix = (int)(runParamsG.Nx / 2.0 + photon.x / runParamsG.dx);
                iy = (int)(runParamsG.Ny / 2.0 + photon.y / runParamsG.dy);
                iz = (int)(photon.z / runParamsG.dz);
                if ((photon.z < runParamsG.zsurf) && (surfflag == 1)) {
                    surfflag = 0;
                    Rd[index] += photon.W;
                    int i2d = (long)(ix*runParamsG.Ny + iy);
                    Rd2D[i2d] += photon.W;
                }
                if (photon.z < 0)
                {
                    photon.photon_status = DEAD;
                    photon.sleft = 0.0;
                }

                bflag = 1;  // Boundary flag. Initialize as 1 = inside volume, then check.
                if (runParamsG.boundaryflag == 0) { // Infinite medium.
                            // Check if photon has wandered outside volume.
                            // If so, set tissue type to boundary value, but let photon wander.
                            // Set blag to zero, so DROP does not deposit energy.
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; bflag = 0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; bflag = 0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; bflag = 0;}
                    if (iz<0)   {iz=0;    bflag = 0;}
                    if (ix<0)   {ix=0;    bflag = 0;}
                    if (iy<0)   {iy=0;    bflag = 0;}
                }
                else if (runParamsG.boundaryflag==1) { // Escape at boundaries
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iz<0)   {iz=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix<0)   {ix=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iy<0)   {iy=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                }
                else if (runParamsG.boundaryflag==2) { // Escape at top surface, no x,y bottom z boundaries
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; bflag = 0.0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; bflag = 0.0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; bflag = 0.0;}
                    if (iz<0)   {iz=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix<0)   {ix=0;    bflag = 0;}
                    if (iy<0)   {iy=0;    bflag = 0;}
                }

                // update pointer to tissue type
                i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
                photon.tiss_type = V[i];
                mua = tissParamsG.muav[photon.tiss_type];
                mus = tissParamsG.musv[photon.tiss_type];
                /* TT parameters af bf ab gb --> q1f,q2f,q3f, q1b,q2b,q3b */
                float gf = tissParamsG.gf[photon.tiss_type];
                float af = tissParamsG.af[photon.tiss_type];
                float gb = tissParamsG.gb[photon.tiss_type];
                float ab = tissParamsG.ab[photon.tiss_type];
                
                q1f = (MC_ONE + gf * gf);
                q2f = pow(MC_ONE - gf, MC_TWO * af);
                q3f = pow(MC_ONE + gf, MC_TWO * af);
                q1b = (MC_ONE + gb * gb);
                q2b = pow(MC_ONE - gb, MC_TWO * ab);
                q3b = pow(MC_ONE + gb, MC_TWO * ab);

            } //(sv) /* same voxel */
          curr_depth++;
        //  photon.num_scatt++;
       } while (curr_depth <= RAY_DEPTH && photon.sleft > 0.0); //do...while

        
        /**** SPIN
        Scatter photon into new trajectory defined by theta and psi.
        Theta is specified by cos(theta), which is determined
        based on the Henyey-Greenstein scattering function.
        Convert theta and psi into cosines ux, uy, uz.
        *****/
        
        float gf = tissParamsG.gf[photon.tiss_type];
        float af = tissParamsG.af[photon.tiss_type];
        float gb = tissParamsG.gb[photon.tiss_type];
        float ab = tissParamsG.ab[photon.tiss_type];
        
        /* two-term sampling */
        rnd = RandomNum;
        if (rnd <= tissParamsG.CC[photon.tiss_type]){ /*more common event */
            rnd = RandomNum;
            photon.costheta = (q1f / (MC_TWO * gf)) - pow(rnd/q2f + (MC_ONE-rnd)/q3f, -MC_ONE/af) / (MC_TWO * gf);
           }
        else{ /* more rare event: backward scatter */
            rnd = RandomNum;
            photon.costheta = (q1b / (MC_TWO * gb)) - pow(rnd/q2b + (MC_ONE-rnd)/q3b, -MC_ONE/ab) / (MC_TWO * gb);
            photon.costheta *= -MC_ONE;
           }
        if (photon.costheta >= MC_ONE) photon.costheta = 1.0-1e-9;
        if (photon.costheta <=-MC_ONE) photon.costheta = -(1.0-1e-9);
        photon.sintheta = sqrt(MC_ONE - photon.costheta*photon.costheta); /*sqrt faster than sin()*/
        
        /* Sample psi. */
        photon.psi = 2.0*PI*RandomNum;
        photon.cospsi = cos(photon.psi);
        if (photon.psi < PI)
            photon.sinpsi = sqrt(1.0 - photon.cospsi*photon.cospsi);     /* sqrt() is faster than sin(). */
        else
            photon.sinpsi = -sqrt(1.0 - photon.cospsi*photon.cospsi);

        /* New trajectory. */
        if (1.0 - fabs(photon.uz) <= ONE_MINUS_COSZERO) {      /* close to perpendicular. */
            photon.uxx = photon.sintheta * photon.cospsi;
            photon.uyy = photon.sintheta * photon.sinpsi;
            photon.uzz = photon.costheta * SIGN(photon.uz);   /* SIGN() is faster than division. */
        }
        else {              /* usually use this option */
            float temp = sqrt(1.0 - photon.uz * photon.uz);
            photon.uxx = photon.sintheta * (photon.ux * photon.uz * photon.cospsi - photon.uy * photon.sinpsi) / temp + photon.ux * photon.costheta;
            photon.uyy = photon.sintheta * (photon.uy * photon.uz * photon.cospsi + photon.ux * photon.sinpsi) / temp + photon.uy * photon.costheta;
            photon.uzz = -photon.sintheta * photon.cospsi * temp + photon.uz * photon.costheta;
        }

        /* Update trajectory */
        photon.ux = photon.uxx;
        photon.uy = photon.uyy;
        photon.uz = photon.uzz;
        photon.num_scatt++;
            
        /**** CHECK ROULETTE
        If photon weight below THRESHOLD, then terminate photon using Roulette technique.
        Photon has CHANCE probability of having its weight increased by factor of 1/CHANCE,
        and 1-CHANCE probability of terminating.
        *****/
        if (photon.W < THRESHOLD) {
            if (RandomNum <= CHANCE)
                photon.W /= CHANCE;
            else photon.photon_status = DEAD;
        }
            
        // Russian roulette with a semi-automatic
        if (photon.W < MIN_VALUE || photon.num_scatt > MAX_SCATT)
            photon.photon_status = DEAD;

        } while (photon.photon_status == ALIVE);  /* end STEP_CHECK_HOP_SPIN */
        /* if ALIVE, continue propagating */
        /* If photon DEAD, then launch new photon. */
}



kernel void MCPolCbsElectricFiedsKernel(
                    device RunParams* run_params,
                    device TissueParams* tissParams,
                    device char* V,
                    device float* Pol3D,
                    device float* Rd,
                    device atomic_float* Rd2D,
                    device atomic_int* DetPhot2D,
                    device atomic_float* Pol2DXX,
                    device atomic_float* Pol2DXY,
                    device atomic_float* Pol2DYX,
                    device atomic_float* Pol2DYY,
                    device atomic_float* Pol2DPhase0,
                    device atomic_float* Pol2DPhase1,
                    device atomic_float* Pol2DPhase2,
                    device atomic_float* Pol2DPhase3,
                    device float* PolVsScatt,
                    device float* PhotonCoordinates,
              //      device float* PolInt02D,
                    device float* PolPhase02D,
                    device SPECKLE* SpeckleData,
                    device atomic_int* SpeckleCount,
                    device int* iRunNum,
                    device float* Intensity_rad,
                    device float* Intensity_azim,
                    device float* Phase_rad,
                    device float* Phase_azim,
                    uint index [[thread_position_in_grid]])
{

 
}


kernel void MCPolCbsElectricFiedsMajoranaKernel(
                    device RunParams* run_params,
                    device TissueParams* tissParams,
                    device char* V,
                    device float* Pol3D,
                    device float* Rd,
                    device atomic_float* Rd2D,
                    device atomic_int* DetPhot2D,
                    device atomic_float* Pol2DXX,
                    device atomic_float* Pol2DXY,
                    device atomic_float* Pol2DYX,
                    device atomic_float* Pol2DYY,
                    device atomic_float* Pol2DPhase0,
                    device atomic_float* Pol2DPhase1,
                    device atomic_float* Pol2DPhase2,
                    device atomic_float* Pol2DPhase3,
                    device float* PolVsScatt,
                    device float* PhotonCoordinates,
              //      device float* PolInt02D,
                    device float* PolPhase02D,
                    device SPECKLE* SpeckleData,
                    device atomic_int* SpeckleCount,
                    device int* iRunNum,
                    device float* Intensity_rad,
                    device float* Intensity_azim,
                    device float* Phase_rad,
                    device float* Phase_azim,
                    uint index [[thread_position_in_grid]])
{


}

kernel void MCRamanKernel(
                       device RunParams* run_params,
                       device TissueParams* tissParams,
                       device char* V,
                       device RadiationData* Rd2D,
                       device PhotonData* Output,
                       device atomic_float* PhotonCoordinates,
                       device atomic_float* PhotonDirections,
                       device atomic_int* PhotonType,
                       device int* iRunNum,
                       uint index [[thread_position_in_grid]])
{
    // SET MAIN PARAMETERS
    int numRuns = *iRunNum;
    Photon photon;
    RunParams runParamsG = *run_params;
    TissueParams tissParamsG = *tissParams;
    RandomGen rng_gen = RandomGen(
        (numRuns ^ (index * 2654435761U)) + 12345U,  // Mixed with golden ratio prime
        (numRuns * 1103515245U) ^ (index * 1013904223U),  // LCG scrambling
        (index ^ numRuns) * 1664525U + 1013904223U // XOR-mix + LCG-style shift
    );

    float width =                       runParamsG.width;    // mm
    float n =                           tissParamsG.n;       // index_of_refraction [-]
    float g =                           tissParamsG.gv[1];   // anisotropy factor [-]
    float r_s =                         1/tissParamsG.musv[1];   // scattering mean free path [mm]
    float r_a =                         1/tissParamsG.muav[1];   // absorption mean free path [mm]
    float step_size =                   runParamsG.step_size;  // mm
    float raman_prob =                  tissParamsG.raman_prob;
    float stim_raman_prob =             tissParamsG.stim_raman_prob;
    float interaction_distance =        tissParamsG.interaction_distance;   // mm wow!
    float laser_beam_radius =           runParamsG.laser_beam_radius;   // mm
    float laser_beam_pulse_width =      runParamsG.laser_beam_pulse_width;
    float laser_beam_pulse_delay =      runParamsG.laser_beam_pulse_delay;   // ps
    float cutoff_radius =               runParamsG.cutoff_radius;    // mm
    float dx =                          2*width/runParamsG.Nx;
    float dy =                          2*width/runParamsG.Ny;
    float dz =                          width/runParamsG.Nz;
    
    float stim_prob =                   stim_raman_prob*step_size;
    int N_steps =                       N_STEPS; //(int)( ceil( C_RAMAN*time_simulated/(index_of_refraction)/r_s)); //step_size
    
    // Initialisation Laser Photons
    /*
    PHOTON TYPE TIPS
    0 - absorbed photon
    1 - laser photon in region
    2 - spontaneous Raman photon in region
    3 - laser photon out of region
    4 - probe photon before entering region
    */

    // Gaussian profile factor for laser
    float sigma = 0.42466090014400953*laser_beam_radius;
    float R = sqrt(-2.0 * sigma * sigma * log(1.0 - RandomNum));
    float theta = 2.0 * PI * RandomNum;
    photon.x = R * cos(theta);
    photon.y = R * sin(theta);

    sigma = 0.42466090014400953*laser_beam_pulse_width*C_RAMAN;    // 1/( 2*sqrt( 2*ln(2) ) )
    R = sqrt( 2.0*sigma*sigma*log( 1.0/(1.0-RandomNum) ) );
    theta = 2.0 * PI * RandomNum;
    photon.z = R*sin( theta ) - laser_beam_pulse_delay*C_RAMAN;

    //photon.ux = 0.0; photon.uy = 0.0; photon.uz = 1.0;
    // set trajectory toward focus
    float r        = 0.1*laser_beam_radius*sqrt(RandomNum); // radius of beam at focus
    float phi        = RandomNum*2.0*PI;
    float xfocus    = r*cos(phi);
    float yfocus    = r*sin(phi);
    float temp    = sqrt((photon.x - xfocus)*(photon.x - xfocus) + (photon.y - yfocus)*(photon.y - yfocus) + runParamsG.zf*runParamsG.zf);
    photon.ux        = -(photon.x - xfocus)/temp;
    photon.uy        = -(photon.y - yfocus)/temp;
    photon.uz        = sqrt(1.0 - photon.ux*photon.ux - photon.uy*photon.uy);
     
    photon.type = 3;
    //photon.type = (index % 100 == 0) ? 4 : 3;

    photon.W = 1.0;
    //bool isInRegion = photon.z >= 0.0f && photon.z <= width;

    // Initialisation Simulation Variables
    float P_Elastic = 1.0 - exp(-step_size/r_s);
    float P_Raman = 1.0 - exp(-raman_prob*step_size);
    float P_Abs = 1.0 - exp(-step_size/r_a);
    //float P_SRS = 1.0 - exp(-stim_raman_prob*step_size);

    //float path = 0.0;
    float ux_old = photon.ux;
    float uy_old = photon.uy;
    float uz_old = photon.uz;
    float x_old = photon.x;
    float y_old = photon.y;
    float z_old = photon.z;
    int photon_type_old = photon.type;

    int Nscatt = 0;
    int srs_flag = 0;
    int data_ID = 3*index*N_steps + Nscatt;
    int data_ID_type = index*N_steps + Nscatt;

    atomic_store_explicit(&PhotonCoordinates[data_ID + 0], photon.x, memory_order_relaxed);
    atomic_store_explicit(&PhotonCoordinates[data_ID + 1], photon.y, memory_order_relaxed);
    atomic_store_explicit(&PhotonCoordinates[data_ID + 2], photon.z, memory_order_relaxed);
    atomic_store_explicit(&PhotonDirections[data_ID + 0], photon.ux, memory_order_relaxed);
    atomic_store_explicit(&PhotonDirections[data_ID + 1], photon.uy, memory_order_relaxed);
    atomic_store_explicit(&PhotonDirections[data_ID + 2], photon.uz, memory_order_relaxed);
    atomic_store_explicit(&PhotonType[data_ID_type], photon.type, memory_order_relaxed);
    
    // MAIN PART
    for(int iPosRuns = 0; iPosRuns < N_steps; iPosRuns++ )
    {

        float vdt; //temp,  phi, theta, c_theta, s_theta, c_phi, s_phi, xi, prob, x_end, y_end
        float t =  iPosRuns*n*step_size/C_RAMAN;

         // ABSORPTION
        #if(ABSORPTION == YES)
         if((photon_type_old == 1 || photon_type_old == 2) && RandomNum < P_Abs)
         {
             photon.type = 0;
             
             // write when absorbed for rendering purposes
             int render_ID = (int)(index*N_steps + iPosRuns);
             Output[render_ID].marker = index + (numRuns - 1) * RAMAN_BATCH;
             Output[render_ID].t = t;
             Output[render_ID].x = photon.x;
             Output[render_ID].y = photon.y;
             Output[render_ID].z = photon.z;
             Output[render_ID].type = photon.type;
             Output[render_ID].W = photon.W;
             
             return;
         }
        #endif
         
         // PROPAGATION
        if (photon.type == 0) return;

         switch (photon_type_old) {
             case 1:  // Laser photon in region
                 vdt = -log(RandomNum)*step_size;     //step_size;

         #if(RAMAN == YES)
                 if (RandomNum < P_Raman) {
                     photon.type = 2;
                     float theta = PI * RandomNum;
                     float phi = 2.0f * PI * RandomNum;  // Use second sample
                     photon.ux = sin(theta) * cos(phi);
                     photon.uy = sin(theta) * sin(phi);
                     photon.uz = cos(theta);
//                     photon.creation_point = float3(x_old, y_old, z_old);

                 }
                 
         #if(SRS == YES)
                 if (photon.type != 2) // Ensure this photon has not already undergone spontaneous Raman
                 {
                     for( int j = 0; j < RAMAN_BATCH; j++ )
                     {
                         int data_ID_temp = 3*(j*N_steps + iPosRuns);
                         int data_ID_type_temp = j*N_steps + iPosRuns;
                         
                         int temp_photon_type = atomic_load_explicit(&PhotonType[data_ID_type_temp], memory_order_relaxed);
                         
                         if (temp_photon_type == 2)
                         {
                             float dx = atomic_load_explicit(&PhotonCoordinates[data_ID_temp + 0], memory_order_relaxed) - x_old;
                             float dy = atomic_load_explicit(&PhotonCoordinates[data_ID_temp + 1], memory_order_relaxed)  - y_old;
                             float dz = atomic_load_explicit(&PhotonCoordinates[data_ID_temp + 2], memory_order_relaxed)  - z_old;
                             if(dx*dx + dy*dy + dz*dz < interaction_distance*interaction_distance)
                             {
                                 if( RandomNum < stim_prob )
                                 {
                                     photon.type = 2;
                                     srs_flag = 1;
                                     atomic_store_explicit(&PhotonType[data_ID_type], 2, memory_order_relaxed);
                                     // SRS photons take the direction of Raman photon
                                     photon.ux = atomic_load_explicit(&PhotonDirections[data_ID_temp + 0], memory_order_relaxed);
                                     photon.uy = atomic_load_explicit(&PhotonDirections[data_ID_temp + 1], memory_order_relaxed);
                                     photon.uz = atomic_load_explicit(&PhotonDirections[data_ID_temp + 2], memory_order_relaxed);
//                                     photon.creation_point = float3(x_old,y_old,z_old);

                                     break;
                                 }
                             }
                         }
                     }
                 }
         #endif
         #endif

         #if(ELASTIC == YES)
                 if (photon.type != 2 && RandomNum < P_Elastic) {
                     updatePhotonDirection(photon, g, {ux_old, uy_old, uz_old}, rng_gen);
                 }
         #endif
                 updatePhotonPosition(photon, vdt, {x_old, y_old, z_old});
                 
                 photon.absorb = photon.W * (1.0 - exp(-vdt/r_a));
                 photon.W -= photon.absorb;
                 
                 break;
             case 2:  // Raman photon in region
                 vdt =  -log(RandomNum)*step_size;     //step_size;

         #if(ELASTIC == YES)
                 if (RandomNum < P_Elastic) {
                     updatePhotonDirection(photon, g, {ux_old, uy_old, uz_old}, rng_gen);
                 }
         #endif
                 updatePhotonPosition(photon, vdt, {x_old, y_old, z_old});
                 
                 photon.absorb = photon.W * (1.0 - exp(-vdt/r_a));
                 photon.W -= photon.absorb;
                 
                 break;
             case 3:  // Laser photon before entering
             case 4:  // Probe photon before entering
                 vdt = n * step_size;
                 updatePhotonPosition(photon, vdt, {x_old, y_old, z_old});

                 if (photon.z >= 0.0) {
                     vdt = -z_old / uz_old + step_size + z_old / (uz_old * n); // ?
                     photon.type = (photon_type_old == 3) ? 1 : 2;
                     updatePhotonPosition(photon, vdt, {x_old, y_old, z_old});
                     photon.entry_time = t;
                     
                     // write when entering region for rendering purposes
                     int render_ID = (int)(index*N_steps + iPosRuns);
                     Output[render_ID].marker = index + (numRuns - 1) * RAMAN_BATCH;
                     Output[render_ID].t = t;
                     Output[render_ID].x = photon.x;
                     Output[render_ID].y = photon.y;
                     Output[render_ID].z = photon.z;
                     Output[render_ID].type = photon.type;
                     Output[render_ID].W = photon.W;
                 }

                 break;
         }
         
         // BOUNDARIES CHECK
         if (photon.type == 1 || photon.type == 2) // Make sure photon is from region
         {
             if (photon.z <= 0) {

                 //float boundary_z = (photon.z >= width) ? width : 0.0;
                 float boundary_z = (photon.z <= 0.0) ? 0.0 : width;
                 vdt = (boundary_z - z_old) / photon.uz;
                 updatePhotonPosition(photon, vdt, {x_old, y_old, z_old});

                 //float rsp = Fresnel::RFresnel(n, 1.0, fabs(uz_old), &photon.uz); //photon.uz);
                 float uz_temp = MC_ZERO;
                 float rsp = Fresnel::RFresnel(n, 1.0, fabs(photon.uz), &uz_temp);
                 //photon.uz = uz_temp;
 
                 if (rsp < MC_ONE)
                 {
                     if (runParamsG.det_state == 0)
                     {
                         int ix = (int)(runParamsG.Nx / 2.0 + photon.x / dx);
                         int iy = (int)(runParamsG.Ny / 2.0 + photon.y / dy);
//                         if (ix >= runParamsG.Nx) ix = runParamsG.Nx - 1;
//                         if (iy >= runParamsG.Ny) iy = runParamsG.Ny - 1;
//                         if (ix<0)   ix = 0;
//                         if (iy<0)   iy = 0;
                         
                         if (ix >= 0 && ix < runParamsG.Nx && iy >= 0 && iy < runParamsG.Ny)
                         {
                             int i2d = (int)(ix*runParamsG.Ny + iy);
                             if (fabs(photon.uz) >= sqrt(1 - runParamsG.NA * runParamsG.NA))
                             {
                                 if (photon.type == 1)
                                     atomic_fetch_add_explicit(
                                                               (device atomic_float*)&Rd2D[i2d].laserRd,
                                                               photon.W * (MC_ONE - rsp),memory_order_relaxed);
                                 if (photon.type == 2)
                                     atomic_fetch_add_explicit(
                                                               (device atomic_float*)&Rd2D[i2d].ramanRd,
                                                               photon.W * (MC_ONE - rsp),memory_order_relaxed);
                                 if (photon.type == 2 && srs_flag == 1)
                                     atomic_fetch_add_explicit(
                                                               (device atomic_float*)&Rd2D[i2d].srsRd,
                                                               photon.W * (MC_ONE - rsp),memory_order_relaxed);
                                     
        //                             if (photon.type == 1) Rd2D[i2d].laserRd += photon.W * (MC_ONE - rsp);
        //                             if (photon.type == 2) Rd2D[i2d].ramanRd += photon.W * (MC_ONE - rsp);
        //                             if (photon.type == 2 & srs_flag == 1) Rd2D[i2d].srsRd += photon.W * (MC_ONE - rsp);
        //                             photon.uz = -photon.uz;
                                 photon.W = photon.W * rsp;
                             }
                         }
                     }

                     photon.type = 0;
                     photon.exit_time = t;
                     
                     // write when exiting region for rendering purposes
                     int render_ID = (int)(index*N_steps + iPosRuns);
                     Output[render_ID].marker = index + (numRuns - 1) * RAMAN_BATCH;
                     Output[render_ID].t = t;
                     Output[render_ID].x = photon.x;
                     Output[render_ID].y = photon.y;
                     Output[render_ID].z = photon.z;
                     Output[render_ID].type = photon.type;
                     Output[render_ID].W = photon.W;
                 }
                 else // internally reflect
                 {
                     photon.uz = -photon.uz;
                     photon.W = photon.W * rsp;
                 }
             }

             if (photon.z >= width) {
                 
                 
     #if (DETECTFORWARD == YES)
                 float boundary_z = (photon.z >= width) ? width : 0.0;
                 vdt = (boundary_z - z_old) / photon.uz;
                 updatePhotonPosition(photon, vdt, {x_old, y_old, z_old});
    
                 //float rsp = Fresnel::RFresnel(n, 1.0, fabs(uz_old), &photon.uz); //photon.uz);
                 float uz_temp = MC_ZERO;
                 float rsp = Fresnel::RFresnel(n, 1.0, fabs(photon.uz), &uz_temp);
//                 photon.uz = uz_temp;

//                     if (photon.type == 1) Rd2D[i2d] += rsp;
                 if (rsp < MC_ONE)
                 {
                     if (runParamsG.det_state == 1)
                     {
                         int ix = (int)(runParamsG.Nx / 2.0 + photon.x / dx);
                         int iy = (int)(runParamsG.Ny / 2.0 + photon.y / dy);
//                                 if (ix >= runParamsG.Nx) ix = runParamsG.Nx - 1;
//                                 if (iy >= runParamsG.Ny) iy = runParamsG.Ny - 1;
//                                 if (ix<0)   ix = 0;
//                                 if (iy<0)   iy = 0;
                         
                         if (ix >= 0 && ix < runParamsG.Nx && iy >= 0 && iy < runParamsG.Ny)
                         {
                             int i2d = (int)(ix*runParamsG.Ny + iy);
                             if (fabs(photon.uz) >= sqrt(1 - runParamsG.NA * runParamsG.NA))
                             {
                                 
                                 if (photon.type == 1)
                                     atomic_fetch_add_explicit(
                                                               (device atomic_float*)&Rd2D[i2d].laserRd,
                                                               photon.W * (MC_ONE - rsp),memory_order_relaxed);
                                 if (photon.type == 2)
                                     atomic_fetch_add_explicit(
                                                               (device atomic_float*)&Rd2D[i2d].ramanRd,
                                                               photon.W * (MC_ONE - rsp),memory_order_relaxed);
                                 if (photon.type == 2 && srs_flag == 1)
                                     atomic_fetch_add_explicit(
                                                               (device atomic_float*)&Rd2D[i2d].srsRd,
                                                               photon.W * (MC_ONE - rsp),memory_order_relaxed);
    //                             Rd2D[i2d] += 1.0;
    //                             if (photon.type == 1) Rd2D[i2d].laserRd += photon.W * (MC_ONE - rsp);
    //                             if (photon.type == 2) Rd2D[i2d].ramanRd += photon.W * (MC_ONE - rsp);
    //                             if (photon.type == 2 & srs_flag == 1) Rd2D[i2d].srsRd += photon.W * (MC_ONE - rsp);
    //                             photon.uz = -photon.uz;
                                 photon.W = photon.W * rsp;
                             }
                         }
                     }
                     
                     photon.type = 0;
                     photon.exit_time = t;
                     
                     // write when exiting region for rendering purposes
                     int render_ID = (int)(index*N_steps + iPosRuns);
                     Output[render_ID].marker = index + (numRuns - 1) * RAMAN_BATCH;
                     Output[render_ID].t = t;
                     Output[render_ID].x = photon.x;
                     Output[render_ID].y = photon.y;
                     Output[render_ID].z = photon.z;
                     Output[render_ID].type = photon.type;
                     Output[render_ID].W = photon.W;
                 }
                 else // internally reflect
                 {
                     photon.uz = -photon.uz;
                     photon.W = photon.W * rsp;
                 }
                     
                 
             }
#endif // End PLOT == NO

         #if (SIDEBOUND == YES)
             if (photon.x * photon.x + photon.y * photon.y > cutoff_radius * cutoff_radius) {
                 photon.type = 0;
             }
         #endif
         }
         
         // SYNCHRONISE ALL THREADS, UPDATE VARIABLES
         x_old = photon.x;
         y_old = photon.y;
         z_old = photon.z;
         ux_old = photon.ux;
         uy_old = photon.uy;
         uz_old = photon.uz;
         photon_type_old = photon.type;

         data_ID = 3*(index*N_steps + iPosRuns);
         data_ID_type = index*N_steps + iPosRuns;
         atomic_store_explicit(&PhotonCoordinates[data_ID + 0], photon.x, memory_order_relaxed);
         atomic_store_explicit(&PhotonCoordinates[data_ID + 1], photon.y, memory_order_relaxed);
         atomic_store_explicit(&PhotonCoordinates[data_ID + 2], photon.z, memory_order_relaxed);
         atomic_store_explicit(&PhotonDirections[data_ID + 0], photon.ux, memory_order_relaxed);
         atomic_store_explicit(&PhotonDirections[data_ID + 1], photon.uy, memory_order_relaxed);
         atomic_store_explicit(&PhotonDirections[data_ID + 2], photon.uz, memory_order_relaxed);
         atomic_store_explicit(&PhotonType[data_ID_type], photon.type, memory_order_relaxed);
        
        
        // TRACKING PHOTON TRAJECTORIES WITH GIVEN TIME STEP FOR RENDERING PURPOSES
        bool boundary_check = true;
        //                                (fabs(photon.x) <= runParamsG.width) &&
        //                              (fabs(photon.y) <= runParamsG.width) &&
        //                              ((photon.z <= runParamsG.width) || (photon.z >= 0));
        if (iPosRuns % RENDER_STEP == 0 && (t != photon.exit_time && t != photon.entry_time) && boundary_check)
        {
           int render_ID = (int)(index*N_steps + iPosRuns);
           Output[render_ID].marker = index + (numRuns - 1) * RAMAN_BATCH;
           Output[render_ID].t = t;
           Output[render_ID].x = photon.x;
           Output[render_ID].y = photon.y;
           Output[render_ID].z = photon.z;
           Output[render_ID].type = photon.type;
           Output[render_ID].W = photon.W;
        }
    }
     
    return;
}

