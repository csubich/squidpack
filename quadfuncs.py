'''module quadfuncs: compute normalized functions for quadrature on the sphere'''

import sympy as sp

def tri_taylor(zfunc,NTri):
    (x,y) = sp.symbols('x y')
    z_approx = 0
    for tpow in range(NTri+1):
        for xpow in range(tpow,-1,-1):
            ypow = tpow - xpow
            z_approx += 1/sp.factorial(xpow)*1/sp.factorial(ypow)*\
                       sp.diff(sp.diff(zfunc,x,xpow),y,ypow).subs({x:0, y:0})*(x**xpow)*(y**ypow)
        
    return z_approx

def tri_ortho(func,NTri,r0):
    # Approximate 'func' in terms of polynomials in x and y (up to and including order NTri) over
    # the domain |x|,|y|<=r0, such that the residual is orthogonal to the approximation over the domain.
    
    ep = r0 # Epsilon
    my_approx = 0
    (x,y) = sp.symbols('x y')
    for tpow in range(NTri+1):
        for xpow in range(tpow,-1,-1):
            ypow = tpow-xpow
            local_poly = sp.polys.orthopolys.legendre_poly(xpow,x/ep)*\
                         sp.polys.orthopolys.legendre_poly(ypow,y/ep)
            local_norm = sp.integrate(sp.integrate(local_poly**2,(x,-ep,ep)),(y,-ep,ep))
            local_dot = sp.integrate(sp.integrate(local_poly*func,(x,-ep,ep)),(y,-ep,ep))
            local_term = local_dot*local_poly/local_norm
            func -= local_term # Necessary for numerical stability -- modified Gram-Schmidt process
            my_approx += local_term
            
    return(my_approx)

def build_zfuncs(NSph,NTri,r0):
    # Build a set of functions (z_poly)*z - (z_approx) that:
    # * Span the set of polynomials x^i y^j z, s.t. i+j < NSph
    # * Are mutually orthogonal over |x|,|y|<r0
    # * Are orthogonal to x^i y^j s.t. i+j <= NTri over the same region

    (x,y) = sp.symbols('x y')
    ep = r0 # Numerical radius
    eps = sp.symbols('epsilon',positive=True,real=True)
    # ep = 
    z = sp.sqrt(1-x**2-y**2)
    z_approx = []
    z_phi = []
    z_poly = []

    for tpow in range(NSph):
#         print('Total power %d' % tpow)
        for xpow in range(tpow,-1,-1):
            ypow = tpow-xpow
#             print('x^%d y^%d' % (xpow,ypow))
            my_poly = x**xpow * y**ypow # Polynomial term multiplying z
            myz = my_poly*z
            #my_approx = tri_taylor(myz,NTri) # Approximation to cancel
            my_taylor = tri_taylor(myz,NTri+NSph+1) # Taylor approximation of z, for integration
            my_approx = tri_ortho(my_taylor,NTri,eps).subs({eps:ep})
            my_phi = (my_taylor-my_approx).expand() # z - approx (Taylor series)

            # Make orthogonal to previous polynomials
            for idx in range(len(z_phi)):
                dotp = sp.integrate(my_phi*z_phi[idx],(x,-ep,ep)).integrate((y,-ep,ep))
                # Take the lowest-order term
        #         et = sp.Poly(dotp).ET()
        #         dotp = et[0].as_expr()*et[1]
                my_phi -= dotp*z_phi[idx]/(4*ep**2)
                my_approx -= dotp*z_approx[idx]/(4*ep**2)
                my_poly -= dotp*z_poly[idx]/(4*ep**2)
            # Normalize
            dotp = sp.integrate(my_phi*my_phi,(x,-ep,ep)).integrate((y,-ep,ep))
        #     et = sp.Poly(dotp).ET()
        #     dotp = et[0].as_expr()*et[1]
            my_phi = ((2*ep)/sp.sqrt(dotp)*my_phi).expand()
            my_approx = ((2*ep)/sp.sqrt(dotp)*my_approx).expand()
            my_poly = sp.expand((2*ep)/sp.sqrt(dotp)*my_poly)

            z_phi.append(my_phi)
            z_approx.append(my_approx)
            z_poly.append(my_poly)
        
    return(z_poly,z_approx)

def make_newton(z_poly, z_approx,kernel='numpy',prec=50):
    # Return residual and derivative functions for the computation of
    # (z_poly*z - z_approx) via Newton's method.
    #
    # Expressing phi = zfunc-z_approx as phi + z_approx = zfunc allows 
    # for an implicit, quadratic function involving phi^2 that permits 
    # analytic cancellation.

    phi = sp.symbols('phi')
    (x,y) = sp.symbols('x y')
    z = sp.sqrt(1-x**2-y**2)

    resid = sp.expand((phi + z_approx)**2 - (z_poly*z)**2).collect(phi)
    resid_phi = sp.diff(resid,phi) # dr/dphi, used for newton's method

    resid_func = sp.lambdify((phi,x,y),resid.evalf(n=prec),kernel)
    deriv_func = sp.lambdify((phi,x,y),resid_phi.evalf(n=prec),kernel)

    # Taking phi = P*z - R, phix = Px*z + P*zx - Rx
    # zx = -x / z, giving z*(phix+R) = Px*z - P*x
    # and...
    poly_x = z_poly.diff(x)
    apx_x = z_approx.diff(x)

    phix = sp.symbols('phix')
    rx = sp.expand(z**2*(phix+apx_x)**2 - (poly_x*z**2 - z_poly*x)**2).collect(phix)
    rx_px = sp.diff(rx,phix)

    rx_func = sp.lambdify((phix,x,y),rx.evalf(n=prec),kernel)
    dx_func = sp.lambdify((phix,x,y),rx_px.evalf(n=prec),kernel)

    # The derivation is identical for y
    poly_y = z_poly.diff(y)
    apx_y = z_approx.diff(y)
    phiy = sp.symbols('phiy')
    ry = sp.expand(z**2*(phiy+apx_y)**2 - (poly_y*z**2 - z_poly*y)**2).collect(phiy)
    ry_py = sp.diff(ry,phiy)

    ry_func = sp.lambdify((phiy,x,y),ry.evalf(n=prec),kernel)
    dy_func = sp.lambdify((phiy,x,y),ry_py.evalf(n=prec),kernel)

    return(resid_func, deriv_func, rx_func, dx_func, ry_func, dy_func)


def get_sph_funcs(NSph, NTri, r0, kernel='numpy', debug_funcs = False, prec=50):
    # Return a list of functions with interface:
    # (phi_out, phi_x, phi_y) = func(xin, yin, compute_grad=True)
    
    # This process involves taking a few steps of Newton's method, and we
    # want to hide that from the caller.
    def make_closure(resid,d_phi,rx,dx,ry,dy):
        # Moment functions resid, d_phi, d_x, and d_y are brought in via lexical closure
        def do_newton(xg,yg,compute_grad=True):
            phi = 0*xg
            # Perform three steps of Newton's method to calculate the output phi
            phi -= resid(1e-16,xg,yg)/d_phi(1e-16,xg,yg)
            phi -= resid(phi,xg,yg)/d_phi(phi,xg,yg)
            phi -= resid(phi,xg,yg)/d_phi(phi,xg,yg)
            
            if (compute_grad):
                gradx = rx(1e-16,xg,yg)/dx(1e-16,xg,yg)
                gradx -= rx(gradx,xg,yg)/dx(gradx,xg,yg)
                gradx -= rx(gradx,xg,yg)/dx(gradx,xg,yg)

                grady = ry(1e-16,xg,yg)/dy(1e-16,xg,yg)
                grady -= ry(grady,xg,yg)/dy(grady,xg,yg)
                grady -= ry(grady,xg,yg)/dy(grady,xg,yg)

                return (phi, gradx, grady)
            else:
                return (phi)
        return do_newton
            
    import sympy as sp
    (x,y) = sp.symbols('x y')
    # Construct the functions phi = (poly*z) - approx
    z = sp.sqrt(1-x**2-y**2)
   
    pr0 = sp.Float(r0,prec)
    (z_poly, z_approx) = build_zfuncs(NSph, NTri,pr0)
    
    funcs_out = []
    
    for idx in range(len(z_poly)):
        (resid,d_phi,rx,dx,ry,dy) = make_newton(z_poly[idx],z_approx[idx],kernel,prec)
        funcs_out.append(make_closure(resid,d_phi,rx,dx,ry,dy))
        
    if (not debug_funcs):
        return funcs_out
    else:
        return (funcs_out, z_poly, z_approx)

def get_tri_funcs(NTri,r0,kernel='numpy', debug_funcs = False, prec=50):
    # Return a list of functions with the interface:
    # (phi_out, phi_x, phi_y) = func
    import sympy as sp
    (x,y) = sp.symbols('x y')
    
    def make_closure(poly,kernel):
        
        ff = sp.lambdify((x,y),poly.evalf(n=prec),kernel)
        fx = sp.lambdify((x,y),poly.diff(x).evalf(n=prec),kernel)
        fy = sp.lambdify((x,y),poly.diff(y).evalf(n=prec),kernel)
        
        def do_func(xg,yg,compute_grad=True):
            # Add 0*xg to each of the terms to ensure that the output is an array.
            # Otherwise, the shape changes surprisingly when phi or its derivative
            # is a constant
            phi = ff(xg,yg)+0*xg
            if (compute_grad):
                dx = fx(xg,yg)+0*xg
                dy = fy(xg,yg)+0*xg
                return (phi,dx,dy)
            else:
                return (phi)
        return do_func
    
    
    funcs_out = []
    polys = []
    pr0 = sp.Float(r0,prec)
    for order in range(0,NTri+1):
        for ypow in range(0,order+1):
            xpow = order-ypow
            poly = sp.polys.orthopolys.legendre_poly(ypow,y/pr0)*\
                   sp.polys.orthopolys.legendre_poly(xpow,x/pr0)*\
                   sp.sqrt((2*xpow+1)*(2*ypow+1))
            polys.append(poly)
            funcs_out.append(make_closure(poly,kernel))
            
    if (not debug_funcs):
        return funcs_out
    else:
        return (funcs_out, polys)


