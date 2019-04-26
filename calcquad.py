'''module calcquad: calculate quadrature points for a spherical hexagon in gnomonic projection'''

import mpmath # Arbitrary precision floating point
from mpmath import mpf # Object for above
from mpmath import mp # Control structure

import numpy as np # Numerical arrays
import quadfuncs # Calculation of functions for quadrature

def gauss(N,dps=15):
    '''Calculate 1D Gaussian quadrature points at arbitrary precision.
 
    Re-implementation of 'gauss.m', from Trefethen's Spectral Methods in MATLAB'''
    with mp.workdps(dps):
        beta = mpmath.matrix(N-1,1)
        for i in range(1,N):
            beta[i-1] = 0.5/(1-(mpmath.mpf(2*i)**-2))**0.5
        T = mpmath.matrix(N,N)
        for i in range(N-1):
            T[i+1,i] = beta[i]
            T[i,i+1] = beta[i]
        (eigval,eigvec) = mpmath.eig(T)
        # Sort eigenvalues
        zl = zip(eigval,range(len(eigval)))
        zl.sort()
        x = np.array([ee[0] for ee in zl])
        w = np.array([2*(eigvec[0,ee[1]]**2) for ee in zl])
    if (dps <= 15):
        x = x.astype('double')
        w = w.astype('double')
    return (x,w)

def quadrilateral_quad(xin, yin, xg, wg):
    '''Calculate 2D quadrature rule over a general quadrilateral, with
     corners specified by [(xin(i), yin(i))], as a tensor product of
     the 1D quadrature rule (xg, wg) over the interval [-1,1]'''

    # Form tensor-product quadrature nodes over the square [-1,1]^2
    (ss, tt) = np.meshgrid(xg,xg,indexing='ij')
    (ws, wt) = np.meshgrid(wg,wg,indexing='ij')

    # Convert these to 1D representations
    ww = (ws*wt).flatten() # Weights
    ss = ss.flatten() # Coordinates along s
    tt = tt.flatten() # Coordinates along t

    # Construct shape functions corresponding to a bilinear map, in order to
    # build a coordinate transform x(s,t) and y(s,t)
    c0 = ((1-ss)*(1-tt))/4
    c1 = ((1+ss)*(1-tt))/4
    c2 = ((1+ss)*(1+tt))/4
    c3 = ((1-ss)*(1+tt))/4

    # Construct the bilinear map itself
    xq = xin[0]*c0 + xin[1]*c1 + xin[2]*c2 + xin[3]*c3
    yq = yin[0]*c0 + yin[1]*c1 + yin[2]*c2 + yin[3]*c3

    # To calculate the quadrature weights, we need the Jacobian of this transform:

    x_s = ((-xin[0] + xin[1])*(1-tt) +
           ( xin[2] - xin[3])*(1+tt))/4
    x_t = ((-xin[0] + xin[3])*(1-ss) +
           (-xin[1] + xin[2])*(1+ss))/4
    y_s = ((-yin[0] + yin[1])*(1-tt) +
           ( yin[2] - yin[3])*(1+tt))/4
    y_t = ((-yin[0] + yin[3])*(1-ss) +
           (-yin[1] + yin[2])*(1+ss))/4

    wq = ww*abs(x_s*y_t - x_t*y_s)
    
    return (xq, yq, wq)

def hex_quad(xin, yin, *args):
    ''' Calculates 2D quadrature for a planar hexagon using a tensor product basis.
    This splits the hexagon into two quadrilaterals, uses quadrilateral_quad on
     each half, and concatenates the results'''
    if (len(xin) != 6 or len(yin) != 6):
        raise(ValueError('Not a hexagon'))
    
    (x0, y0, w0) = quadrilateral_quad(xin[[0,1,2,3]],yin[[0,1,2,3]],*args)
    (x1, y1, w1) = quadrilateral_quad(xin[[0,3,4,5]],yin[[0,3,4,5]],*args)
    
    xout = np.concatenate((x0,x1))
    yout = np.concatenate((y0,y1))
    wout = np.concatenate((w0,w1))
    return(xout,yout,wout)

def gnom_weight(xin,yin):
    '''Calculate the weight function for integration over the sphere under
    a gnomonic projection.'''
    
    return (1+xin**2 + yin**2)**-1.5

def eval_moments(xg,yg,funcs):
    ''' Evaluate the moment functions given by `funcs` on the input
    coordinates (xg,yg), specified in a gnomonic projection '''
    
    # Calculate x, y in Cartesian coordinates
    z = (1 + xg**2 + yg**2)**-0.5
    xc = z*xg
    yc = z*yg
    
    # Determine the number of moments we're dealing with
    num = len(funcs)
    
    # Allocate an output variable
    moms_out = np.zeros( (num,) + xg.shape, # Array shape
                          dtype = xg.dtype)
    
    for idx in range(num):
        moms_out[idx,:] = funcs[idx](xc,yc,compute_grad=False)
    
    return(moms_out)

def calc_quad_init(N_init,N_targ,funcs,q1x,q1w,r0,xhex = None, yhex = None):
    '''Calculate an initial quadrature rule.  If not specified, calculate the rule
    for a regular hexagon that inscribes the circle of radius r0'''
    # Calculate an initial quadrature rule for a regular hexagon
    
    # Define the hexagon
    if (xhex is None):
        xhex = r0*np.cos(2*np.pi*np.arange(6)/6)
        yhex = r0*np.sin(2*np.pi*np.arange(6)/6)
    # Calculate a large quadrature rule for this hexagon, to give near-exact quadrature
    # values to match with the reduced rule
    
    (hqx, hqy, hqw) = hex_quad(xhex,yhex,q1x.astype('double'),q1w.astype('double'))
    gnom_w = gnom_weight(hqx, hqy)
    hqw = (hqw*gnom_w).astype('double')
    
    # Evaluate the spherical harmonic moments with this quadrature rule
    moments = eval_moments(hqx,hqy,funcs)
    quad_exact_hp = np.dot(moments,hqw)
    quad_exact = np.double(quad_exact_hp)
    
    # Initialize quadrature points and weights to iteratively optimize
    xlow = 0.6*r0*(2*np.mod(np.arange(N_init)*2**0.5,1)-1)
    ylow = 0.6*r0*(2*np.mod(np.arange(N_init)*3**0.5,1)-1)
    wlow = xlow**0 + 1e-2*(2*np.mod(np.arange(N_init)*5**0.5,1)-1)
    wlow *= quad_exact[0]/np.sum(wlow)
    

    itercount = 0
    loop_err = 1
    N_quad = len(wlow)    
    while (itercount < 200):
        N_quad = len(wlow)    
        # Evaluate moment functions and gradients at candidate quadrature points
        moms = eval_moments(xlow,ylow,funcs)
        drdx = (eval_moments(xlow+1e-3*r0,ylow,funcs) - eval_moments(xlow-1e-3*r0,ylow,funcs))/(2e-3*r0)
        drdy = (eval_moments(xlow,ylow+1e-3*r0,funcs) - eval_moments(xlow,ylow-1e-3*r0,funcs))/(2e-3*r0)

        # Evaluate the quadrature rule to determine residual
        resid = quad_exact - np.dot(moms,wlow)
        # Define scalar error
        loop_err = np.sum(resid**2)**0.5
        if (loop_err < 1e-14*quad_exact[0] and N_quad <= N_targ):
#             print('Done! (%d iterations)' % itercount)
#             print('  Error %e' % loop_err)
            break
#         print('(%d) Error %e, %d/%d points' % (itercount,loop_err, N_quad,N_targ))
#         if (N_quad > N_targ):
#             print('  Minimum weight %e' % np.min(abs(wlow)))

        itercount += 1

        # Dresid/Dw is equal to the moment function
        drdw = moms
        # Build the Jacobian matrix
        Jac = np.concatenate((wlow*drdx,wlow*drdy,drdw),axis=1)

        # Extend Jacobian to remove a point, if N_quad exceeds the target
        if (N_quad > N_targ):
            softmin = sum(abs(wlow)**-1)/sum(wlow**-2) # Soft minimization function
            # Its gradient with respect to weights:
            dsdw = -wlow**(-2)/sum(wlow**-2) - \
                    softmin/sum(wlow**-2)*(-2*wlow**-3)

            # Build the Jacobian row:
            dsdq = np.concatenate((0*dsdw, 0*dsdw, dsdw))
            dsdq = dsdq.reshape((1,3*N_quad))
            Jac = np.concatenate((Jac,dsdq*1e-9/loop_err),axis=0)

            # Add the normalization factor to the residual:
            resid = np.concatenate((resid,[-0.9*softmin*1e-9/loop_err]))


        (delta,lsq_resid,lsq_rank,jac_singval) = np.linalg.lstsq(Jac,resid,rcond=None)
        delta = np.array(delta).reshape((-1,N_quad))

        scale = min(1, 0.99*np.min(np.abs(wlow)/(1e-16+np.abs(delta[2,:]))))
        xlow += scale*delta[0,:]
        ylow += scale*delta[1,:]
        wlow += scale*delta[2,:]

        if (np.min(abs(wlow)) <= 1e-6*quad_exact[0] and N_quad > N_targ):
            drop_pt = np.argmin(abs(wlow))
            xlow = xlow[np.arange(N_quad) != drop_pt]
            ylow = ylow[np.arange(N_quad) != drop_pt]
            wlow = wlow[np.arange(N_quad) != drop_pt]
#             print('  Dropping point %d' % drop_pt)
                
    return (xlow,ylow,wlow,loop_err,itercount,quad_exact)

def calc_hex_quad(xhex,yhex,xreg,yreg,wreg,quad_reg,funcs,q1x,q1w,r0,debug_print = False):
    '''Calculate a quadrature rule for an irregular spherical hexagon (coordinates `xhex` and `yhex`)
    by adapting a pregenerated quadrature rule (`xreg`, `yreg`, `wreg`, s.t. M(xreg,yreg)*wreg = `quad_reg`)'''
    
    # Calculate our goal quadrature values
    (hqx, hqy, hqw) = hex_quad(xhex,yhex,q1x.astype('double'),q1w.astype('double'))
    gnom_w = gnom_weight(hqx, hqy)
    hqw = (hqw*gnom_w).astype('double')

    # Evaluate the spherical harmonic moments with this quadrature rule
    moments = eval_moments(hqx,hqy,funcs)
    quad_exact = np.dot(moments,hqw)
    quad_diff = quad_exact - quad_reg
    qdn = (np.sum(quad_diff**2))**0.5 # Norm of the difference, used to normalize the transition

    # Initialize candidate quadrature points and weights
    xlow = xreg.copy()
    ylow = yreg.copy()
    wlow = wreg.copy()
    theta = 1.0 # Transition parameter -- 0 uses quad_exact
    N_quad = len(xlow)
    # Corresponding row to add to the Jacobian matrix
    indic = qdn*(np.arange(3*N_quad+1) == (3*N_quad)).astype('double').reshape(1,3*N_quad+1)
    itercount = 0
    
    while (itercount < 100):
        moms = eval_moments(xlow,ylow,funcs)
        drdx = (eval_moments(xlow+1e-3*r0,ylow,funcs) - eval_moments(xlow-1e-3*r0,ylow,funcs))/(2e-3*r0)
        drdy = (eval_moments(xlow,ylow+1e-3*r0,funcs) - eval_moments(xlow,ylow-1e-3*r0,funcs))/(2e-3*r0)

        # Evaluate the quadrature rule to determine residual
        resid = quad_exact - np.dot(moms,wlow) - theta*quad_diff
        loop_err = np.sqrt(np.sum(resid**2) + (qdn*theta)**2)
        if (debug_print):
            print('%d: %.3e (%.3e%+.3e)' % (itercount,loop_err,np.sqrt(sum(resid**2)),qdn*theta))

        if (loop_err < 1e-14*quad_reg[0]*N_quad**0.5):
            if (debug_print):
                print('Done!')
                print('  %d iterations' % itercount)
                print('  Error %.3e' % loop_err)
            break
        itercount += 1
        # Dresid/Dw is equal to the moment function
        drdw = moms
        # Build the Jacobian matrix
        Jac = np.concatenate((wlow*drdx,
                              wlow*drdy,
                              drdw,
                              quad_diff.reshape(quad_exact.shape[0],1)
                             ),axis=1)
        Jac = np.concatenate((Jac,indic),axis=0)
        resid = np.concatenate((resid,[-qdn*theta]))
        (delta,lsq_resid,lsq_rank,jac_singval) = np.linalg.lstsq(Jac,resid,rcond=None)
        deltheta = delta[-1]
        delta = delta[:-1].reshape((-1,N_quad))

        scale = min(1, 0.5*np.min(np.abs(wlow)/(1e-16+np.abs(delta[2,:]))))
        if (debug_print): print('   Scale %.3e' % scale)
        xlow += scale*delta[0,:]
        ylow += scale*delta[1,:]
        wlow += scale*delta[2,:]
        theta += scale*deltheta
        
    return(xlow,ylow,wlow,loop_err,itercount)

def quadone(verts,xreg,yreg,wreg,quad_reg,funcs,q1x,q1w,r0,debug_print=False):
    '''quadone: Calculate quadrature rule for a single spherical hexagon/pentagon'''
    
    # Project the vertices into a tangent plane via the gnomonic projection.  Call these coordinates {r,s,t}

    # Calculate an approximate center of the cell, by averaging the vertices
    center = np.mean(verts,axis=0)
    # Project back to the unit sphere
    center /= np.linalg.norm(center)

    if (verts.shape[0] == 5):
        # We have a hexagon, not a pentagon, so duplicate a vertex
        verts = verts[[0]+range(5),:]

    t_vec = center # t is the outward coordinate, which takes the place of 'z'
    # Pick one of the vertex directions as 'r' (x-substitute)
    r_vec = verts[0,:] - center
    r_vec -= t_vec*np.dot(r_vec,t_vec)
    r_vec /= np.linalg.norm(r_vec)

    # Pick the other direction as 's' (y-substitute)
    s_vec = np.cross(t_vec,r_vec)

    # These vectors give a transform matrix
    tmat = np.array([r_vec,s_vec,t_vec]) # Convert x/y/z to r/s/t
    tmati = np.linalg.inv(tmat) # Convert r/s/t to x/y/z

    # Convert the hexagon vertices to {r,s,t} space
    proj_verts = np.dot(tmat,verts.T)

    # Divide the coordinates by their third component to effect the gnomonic projection
    hex_gr = proj_verts[0,:]/proj_verts[2,:]
    hex_gs = proj_verts[1,:]/proj_verts[2,:]

    # Adjust the regular quadrature rule to this particular polygon
    (quad_gr,quad_gs,quad_w,loop_err,itercount) = calc_hex_quad(hex_gr,hex_gs,
                                                                xreg,yreg,wreg,quad_reg,
                                                                funcs,q1x,q1w,r0,debug_print=debug_print)

    rel_err = loop_err/quad_reg[0]
    
    # Convert the quadrature points on the gnonomic (r,s) plane back to a rotated sphere:

    # First, place the points on the tangent plane
    quad_verts = np.dot(tmat.T,np.vstack((quad_gr, quad_gs, quad_gr**0)))

    # Then, normalize to the unit sphere
    quad_verts = quad_verts / np.sqrt(np.sum(quad_verts**2,axis=0))
    quad_verts = quad_verts.T

    return(quad_verts,quad_w,rel_err,itercount)


def quadglb(verts):
    ''' Helper (currying) function for quadone, to use the global namespace for all static parameters '''
    return quadone(verts,xreg,yreg,wreg,quad_reg,funcs,q1x,q1w,r0)

def quadgrid(primal_vertices,vert_locs,q1x,q1w,NTri,NSph,N_start,N_targ):
    '''quadgrid: calculate quadrature rule for each element of a hexagonal/pentagonal grid'''
    
    # Calculate the grid scale factor
    r0 = (4.0/len(primal_vertices))**0.5

    print('Scale factor %.2e for %d cells' % (r0,len(primal_vertices)))

    # Get approximately-orthogonal moment functions
    print('  Computing moment functions for order %d x %d' % (NTri,NSph))
    (funcs_sph) = quadfuncs.get_sph_funcs(NSph,NTri,r0,debug_funcs=False)
    funcs_tri = quadfuncs.get_tri_funcs(NTri,r0)
    funcs = funcs_tri + funcs_sph
    print('  ... complete')
    
    # Get initial quadrature rule for a regular hexagon
    (xreg,yreg,wreg,loop_err,itercount,quad_reg) = calc_quad_init(N_start,N_targ,funcs,q1x,q1w,r0)

    print('  Found quadrature rule with %.4e residual error, %d iterations' % (loop_err/quad_reg[0], itercount))


    # Plot said rule
#     plt.figure()
#     plt.plot(np.cos(2*np.pi*np.arange(7)/6),np.sin(2*np.pi*np.arange(7)/6),'k-')
#     plt.plot(np.cos(2*np.pi*np.arange(65)/64),np.sin(2*np.pi*np.arange(65)/64),'b-')
#     plt.plot(xreg/r0,yreg/r0,'r.')
#     plt.title('Initial quadrature rule')
#     plt.show()
    
    verts = [vert_locs[primal_vertices[vidx],:] for vidx in range(len(primal_vertices))]

    try:
        # Compute quadratures in parallel via ipyparallel, if available
        import ipyparallel
        #raise ImportError
        rc = ipyparallel.Client()
        view = rc[:]
        view.push(dict(xreg=xreg,yreg=yreg,wreg=wreg,quad_reg=quad_reg,
                       funcs=funcs,q1x=q1x,q1w=q1w,r0=r0,quadone=quadone))

        print('  Processing in parallel')
        # ipyparallel.util.interactive is necessary to 'push' everything to the global namespace
        # on the client processes
        quad_outs = view.map_sync(ipyparallel.util.interactive(quadglb),verts)

        rc.close()
    except ImportError: # Ipyparallel not available
        print('  Processing in serial')
        quad_outs = map(lambda (vv) : quadone(vv,xreg,yreg,wreg,quad_reg,funcs,q1x,q1w,r0),
                        verts)
    (quad_points, quad_weights, rel_errs, iters) = zip(*quad_outs)

    total_iter = sum(iters) 
    print('Total iterations %d (avg %f)' % (total_iter, total_iter*1.0/len(verts)))
    print('Maximum relative error %e' % (max(rel_errs)))
    
    # Return quadrature points and weights
    return (quad_points, quad_weights, rel_errs, total_iter, xreg, yreg, wreg)
