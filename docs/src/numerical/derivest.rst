Introduction
############

The general problem of differentiation of a function typically pops up in three ways in Python.

-  The symbolic derivative of a function.

-  Compute numerical derivatives of a function defined only by a sequence of data points.

-  Compute numerical derivatives of a analytically supplied function.

Clearly the first member of this list is the domain of the symbolic toolbox SymPy, or some set of symbolic tools. Numerical differentiation of a function defined by data points can be achieved with the function gradient, or perhaps by differentiation of a curve fit to the data, perhaps to an interpolating spline or a least squares spline fit.

The third class of differentiation problems is where Numdifftools is valuable. This document will describe the methods used in Numdifftools and in particular the Derivative class.


Numerical differentiation of a general function of one variable
###############################################################

Surely you recall the traditional definition of a derivative, in terms of a limit.

.. math::
    f'(x) = \lim_{\delta \to 0}{\frac{f(x+\delta) - f(x)}{\delta}}
    :label: 1

For small :math:`\delta`, the limit approaches :math:`f'(x)`. This is a one-sided approximation for the derivative. For a fixed value of :math:`\delta`, this is also known as a finite difference approximation (a forward difference.) Other approximations for the derivative are also available. We will see the origin of these approximations in the Taylor series expansion of a function :math:`f(x)` around some point :math:`x_0`.

.. math::
    f(x_0+\delta) &= f(x_0) + \delta f'(x_0) + \frac{\delta^2}{2} f''(x_0) + \frac{\delta^3}{6} f^{(3)}(x_0) + \\
    :label: 2

    & \frac{\delta^4}{24} f^{(4)}(x_0) + \frac{\delta^5}{120} f^{(5)}(x_0) + \frac{\delta^6}{720} f^{(6)}(x_0) +...\\


Truncate the series in :eq:`2` to the first three terms, divide by :math:`\delta` and rearrange yields the forward difference approximation :eq:`1`:

.. math::
    f'(x_0) = \frac{f(x_0+\delta) - f(x_0)}{\delta} - \frac{\delta}{2} f''(x_0) - \frac{\delta^2}{6} f'''(x_0) + ...
    :label: 3

When :math:`\delta` is small, :math:`\delta^2` and any higher powers are vanishingly small. So we tend to ignore those higher powers, and describe the approximation in :eq:`3` as a first order approximation since the error in this approximation approaches zero at the same rate as the first power of :math:`\delta`.  [1]_ The values of :math:`f''(x_0)` and :math:`f'''(x_0)`, while unknown to us, are fixed constants as :math:`\delta` varies.

Higher order approximations arise in the same fashion. The central difference :eq:`4` is a second order approximation.

.. math::
    f'(x_0) = \frac{f(x_0+\delta) - f(x_0-\delta)}{2\delta} - \frac{\delta^2}{3} f'''(x_0) + ...
    :label: 4


Unequally spaced finite difference rules
########################################

While most finite difference rules used to differentiate a function will use equally spaced points, this fails to be appropriate when one does not know the final spacing. Adaptive quadrature rules can succeed by subdividing each sub-interval as necessary. But an adaptive differentiation scheme must work differently, since differentiation is a point estimate. Derivative generates a sequence of sample points that follow a log spacing away from the point in question, then it uses a single rule (generated on the fly) to estimate the desired derivative. Because the points are log spaced, the same rule applies at any scale, with only a scale factor applied.


Odd and even transformations of a function
##########################################
.. index:: odd transformation

Returning to the Taylor series expansion of :math:`f(x)` around some point :math:`x_0`, an even function  [2]_ around :math:`x_0` must have all the odd order derivatives vanish at :math:`x_0`. An odd function has all its even derivatives vanish from its expansion. Consider the derived functions :math:`f_{odd}(x)` and :math:`f_{even}(x)`.

.. math::
    f_{odd}(x) = \frac{f(x_0 + x) - f(x_0 - x )}{2}
    :label: 5

.. math::
    f_{even}(x) = \frac{f(x_0 + x) - 2f(x_0) + f(x_0 - x)}{2}
    :label: 6

The Taylor series expansion of :math:`f_{odd}(x)` around zero has the useful property that we have killed off any even order terms, but the odd order terms are identical to :math:`f(x)`, as expanded around :math:`x_0`.

.. math::
    f_{odd}(\delta) = \delta f'(x_0) + \frac{\delta^3}{6} f^{(3)}(x_0) + \frac{\delta^5}{120} f^{(5)}(x_0) + \frac{\delta^7}{5040} f^{(7)}(x_0) +...
    :label: 7

Likewise, the Taylor series expansion of :math:`f_{even}(x)` has no odd order terms or a constant term, but other even order terms that are identical to :math:`f(x)`.

.. index:: even transformation

.. math::
    f_{even}(\delta) = \frac{\delta^2}{2} f^{(2)}(x_0) + \frac{\delta^4}{24} f^{(4)}(x_0) + \frac{\delta^6}{720} f^{(6)}(x_0) + \frac{\delta^8}{40320} f^{(8)}(x_0) + ...
    :label: 8


The point of these transformations is we can rather simply generate a higher order approximation for any odd order derivatives of :math:`f(x)` by working with :math:`f_{odd}(x)`. Even order derivatives of :math:`f(x)` are similarly generated from :math:`f_{even}(x)`. For example, a second order approximation for :math:`f'(x_0)` is trivially written in :eq:`9` as a function of :math:`\delta`.

.. math::
    f'(x_0; \delta) = \frac{f_{odd}(\delta)}{\delta} - \frac{\delta^2}{6} f^{(3)}(x_0)
    :label: 9

We can do better rather simply, so why not? :eq:`10` shows a fourth order approximation for :math:`f'(x_0)`.

.. math::
    f'(x_0; \delta) = \frac{8 f_{odd}(\delta)-f_{odd}(2\delta)}{6\delta} + \frac{\delta^4}{30} f^{(5)}(x_0)
    :label: 10

Again, the next non-zero term :eq:`11` in that expansion has a higher power of :math:`\delta` on it, so we would normally ignore it since the lowest order neglected term should dominate the behavior for small :math:`\delta`.

.. math::
    \frac{\delta^6}{252} f^{(7)}(x_0)
    :label: 11

Derivative uses similar approximations for all derivatives of :math:`f` up to any order. Of course, it is not always possible for evaluation of a function on both sides of a point, as central difference rules will require. In these cases, you can specify forward or backward difference rules as appropriate. You can also specify to use the complex step derivative, which we will outline in the next section.

Complex step derivative
#######################
The derivation of the complex-step derivative approximation is accomplished by replacing :math:`\delta` in :Eq:`2` 
with a complex step :math:`i h`:

.. math::
    f(x_0+ i h) &= f(x_0) + i h f'(x_0) - \frac{h^2}{2} f''(x_0) - \frac{i h^3}{6} f^{(3)}(x_0) + \frac{h^4}{24} f^{(4)}(x_0) + \\
    :label: 12a

    & \frac{i h^5}{120} f^{(5)}(x_0) - \frac{h^6}{720} f^{(6)}(x_0) -...\\


Taking only the imaginary parts of both sides gives

.. math::
    \, \Im \,(f(x_0+ i h)) &= h f'(x_0)  - \frac{h^3}{6} f^{(3)}(x_0) + \frac{h^5}{120} f^{(5)}(x_0) -...
    :label: 12b

Dividing with :math:`h` and rearranging yields:

.. math::
    f'(x_0) = \Im(f(x_0+ i h))/ h   + \frac{h^2}{6} f^{(3)}(x_0) - \frac{h^4}{120} f^{(5)}(x_0) +...
    :label: 12c

Terms with order :math:`h^2` or higher can safely be ignored since the interval :math:`h` can be chosen up to machine precision
without fear of rounding errors stemming from subtraction (since there are not any). Thus to within second-order the complex-step derivative approximation is given by:

.. math::
    f'(x_0) = \Im(f(x_0 + i h))/ h
    :label: 12d


Next, consider replacing the step :math:`\delta` in :Eq:`8` with the complex step :math:`i^\frac{1}{2}  h`:

.. math::
    \quad f_{even}(i^\frac{1}{2} h) &= \frac{i h^2}{2} f^{(2)}(x_0) - \frac{h^4}{24} f^{(4)}(x_0) - \frac{i h^6}{720} f^{(6)}(x_0) + \\
    :label: 12e

		& \frac{h^8}{40320} f^{(8)}(x_0) + \frac{i h^{10}}{3628800} f^{(10)}(x_0) -...\\

Similarly dividing with :math:`h^2/2` and taking only the imaginary components yields:

.. math::
    \quad f^{(2)}(x_0) = \Im\,(2\,f_{even}(i^\frac{1}{2} h)) / h^2 + \frac{h^4}{360} f^{(6)}(x_0) - \frac{h^8}{1814400} f^{(10)}(x_0)...
    :label: 12f

This approximation is still subject to difference errors, but the error associated with this approximation is proportional to 
:math:`h^4`. Neglecting these higher order terms yields:

.. math::
    \quad f^{(2)}(x_0) = 2 \Im\,(f_{even}(i^\frac{1}{2} h)) / h^2 = \Im(f(x_0 + i^\frac{1}{2} h) + f(x_0-i^\frac{1}{2} h)) / h^2
    :label: 12g

See [LaiCrassidisCheng2005]_ and [Ridout2009]_ for more details.
The complex-step derivative in numdifftools.Derivative has truncation error 
:math:`O(\delta^4)` for both odd and even order derivatives for :math:`n>1`. For :math:`n=1`
the truncation error is on the order of :math:`O(\delta^2)`, so
truncation error can be eliminated by choosing steps to be very small.  The first order complex-step derivative avoids the problem of
round-off error with small steps because there is no subtraction. However,
the function to differentiate needs to be analytic. This method does not work if it does
not support complex numbers or involves non-analytic functions such as
e.g.: abs, max, min. For this reason the `central` method is the default method.
    

High order derivative
#####################
So how do we construct these higher order approximation formulas? Here we will deomonstrate the principle by computing the 6'th order central approximation for the first-order derivative. In order to do so we simply set :math:`f_{odd}(\delta)` equal to its 3-term Taylor expansion:

.. math::
    f_{odd}(\delta) = \sum_{i=0}^{2} \frac{\delta^{2i+1}}{(2i+1)!} f^{(2i+1)}(x_0)
    :label: 12

By inserting three different stepsizes into :eq:`12`, eg :math:`\delta, \delta/2, \delta/4`, we get a set of linear equations:

.. math::
    \begin{bmatrix}
        1 & \frac{1}{3!} & \frac{1}{5!} \\
        \frac{1}{2} & \frac{1}{3! \, 2^3} & \frac{1}{5! \, 2^5} \\
        \frac{1}{4} & \frac{1}{3! \, 4^3} & \frac{1}{5! \, 4^5}
    \end{bmatrix}
    \begin{bmatrix}
        \delta f'(x_0) \\
        \delta^3 f^{(3)}(x_0) \\
        \delta^5 f^{(5)}(x_0)
    \end{bmatrix} =
    \begin{bmatrix}
        f_{odd}(\delta) \\
        f_{odd}(\delta/2) \\
        f_{odd}(\delta/4)
    \end{bmatrix}
    :label: 13

The solution of these equations are simply:

.. math::
    \begin{bmatrix}
        \delta f'(x_0) \\
        \delta^3 f^{(3)}(x_0) \\
        \delta^5 f^{(5)}(x_0)
    \end{bmatrix} = \frac{1}{3}
    \begin{bmatrix}
        \frac{1}{15} & \frac{-8}{3} & \frac{256}{15} \\
        -8 & 272 & -512 \\
        512 & -5120 & 8192
    \end{bmatrix}
    \begin{bmatrix}
        f_{odd}(\delta) \\
        f_{odd}(\delta/2) \\
        f_{odd}(\delta/4)
    \end{bmatrix}
    :label: 14a

The first row of :eq:`14a` gives the coefficients for 6'th order approximation. Looking at at row two and three, we see also that 
this gives the 6'th order approximation for the 3'rd and 5'th order derivatives as bonus. Thus this is also a general method for obtaining high order differentiation rules. As previously noted these formulas have the additional benefit of beeing applicable to any scale, with only a scale factor applied.


Richardson extrapolation methodology applied to derivative estimation
#####################################################################

.. index:: Richardson extrapolation

Some individuals might suggest that the above set of approximations are entirely adequate for any sane person. Can we do better?

Suppose we were to generate several different estimates of the approximation in :eq:`3` for different values of :math:`\delta` at a fixed :math:`x_0`. Thus, choose a single :math:`\delta`, estimate a corresponding resulting approximation to :math:`f'(x_0)`, then do the same for :math:`\delta/2`. If we assume that the error drops off linearly as :math:`\delta \to 0`, then it is a simple matter to extrapolate this process to a zero step size. Our lack of knowledge of :math:`f''(x_0)` is irrelevant. All that matters is :math:`\delta` is small enough that the linear term dominates so we can ignore the quadratic term, therefore the error is purely linear.

.. math::
    f'(x_0) = \frac{f(x_0+\delta) - f(x_0)}{\delta} - \frac{\delta}{2} f''(x_0)
    :label: 15

The linear extrapolant for this interval halving scheme as :math:`\delta \to 0` is given by:

.. math::
    f^{'}_{0} = 2 f^{'}_{\delta/2} - f^{'}_{\delta}
    :label: 16

Since I've always been a big fan of convincing myself that something will work before I proceed too far, lets try this out in Python. Consider the function :math:`e^x`. Generate a pair of approximations to :math:`f'(0)`, once at :math:`\delta` of 0.1, and the second approximation at :math:`1/2` that value. Recall that :math:`\frac{d(e^x)}{dx} = e^x`, so at x = 0, the derivative should be exactly 1. How well will we do?

   >>> from numpy import exp, allclose
   >>> f = exp
   >>> dx = 0.1
   >>> df1 = (f(dx) - f(0))/dx
   >>> allclose(df1, 1.05170918075648)
   True

   >>> df2 = (f(dx/2) - f(0))/(dx/2)
   >>> allclose(df2, 1.02542192752048)
   True

   >>> allclose(2*df2 - df1, 0.999134674284488)
   True


In fact, this worked very nicely, reducing the error to roughly 1 percent of our initial estimates. Should we be surprised at this reduction? Not if we recall that last term in :eq:`3`. We saw there that the next term in the expansion was :math:`O(\delta^2)`. Since :math:`\delta` was 0.1 in our experiment, that 1 percent number makes perfect sense.

The Richardson extrapolant in :eq:`16` assumed a linear process, with a specific reduction in :math:`\delta` by a factor of 2. Assume the two term (linear + quadratic) residual term in :eq:`3`, evaluating our approximation there with a third value of :math:`\delta`. Again, assume the step size is cut in half again. The three term Richardson extrapolant is given by:

.. math::
    f'_0 = \frac{1}{3}f'_\delta - 2f'_{\delta/2} + \frac{8}{3}f'_{\delta/4}
    :label: 14

A quick test in Python yields much better results yet.

    >>> from numpy import exp, allclose
    >>> f = exp
    >>> dx = 0.1

    >>> df1 = (f(dx) - f(0))/dx
    >>> allclose(df1,  1.05170918075648)
    True

    >>> df2 = (f(dx/2) - f(0))/(dx/2)
    >>> allclose(df2, 1.02542192752048)
    True

    >>> df3 = (f(dx/4) - f(0))/(dx/4)
    >>> allclose(df3, 1.01260482097715)
    True

    >>> allclose(1./3*df1 - 2*df2 + 8./3*df3, 1.00000539448361) 
    True

Again, Derivative uses the appropriate multiple term Richardson extrapolants for all derivatives of :math:`f` up to any order [3]_. This, combined with the use of high order approximations for the derivatives, allows the use of quite large step sizes. See [LynessMoler1966]_ and [LynessMoler1969]_. How to compute the multiple term Richardson extrapolants will be elaborated further in the next section.


Multiple term Richardson extrapolants
#####################################

.. index:: Richardson extrapolation

We shall now indicate how we can calculate the multiple term Richardson extrapolant for :math:`f_{odd}(\delta)/\delta` by rearranging :eq:`12`:

.. math::
    \frac{f_{odd}(\delta)}{\delta} = f'(x_0) + \sum_{i=1}^{\infty} \frac{\delta^{2i}}{(2i+1)!} f^{(2i+1)}(x_0)
    :label: 17

This equation has the form

.. math::
    \phi(\delta) = L + a_0 \delta^2 + a_1 \delta^4 + a_2 \delta^6 + ...
    :label: 18

where L stands for :math:`f'(x_0)` and :math:`\phi(\delta)` for the numerical differentiation formula :math:`f_{odd}(\delta)/\delta`.

By neglecting higher order terms (:math:`a_3 \delta^8`) and inserting three different stepsizes into :eq:`18`, eg :math:`\delta, \delta/2, \delta/4`, we get a set of linear equations:

.. math::
    \begin{bmatrix}
        1 & 1 & 1 \\
        1 & \frac{1}{2^2} & \frac{1}{2^4} \\
        1 & \frac{1}{4^2} & \frac{1}{4^4}
    \end{bmatrix}
    \begin{bmatrix}
        L \\
        \delta^2 a_0 \\
        \delta^4 a_1
    \end{bmatrix} =
    \begin{bmatrix}
        \phi(\delta) \\
        \phi(\delta/2) \\
        \phi(\delta/4)
    \end{bmatrix}
    :label: 19

The solution of these equations are simply:

.. math::
    \begin{bmatrix}
        L \\
        \delta^2 a_0 \\
        \delta^4 a_1
    \end{bmatrix} =  \frac{1}{45}
    \begin{bmatrix}
        1 & -20 & 64 \\
        -20 & 340 & -320 \\
        64 & -320 & 256
    \end{bmatrix}
    \begin{bmatrix}
        \phi(\delta) \\
        \phi(\delta/2) \\
        \phi(\delta/4)
    \end{bmatrix}
    :label: 20

The first row of :eq:`20` gives the coefficients for Richardson extrapolation scheme.


Uncertainty estimates for Derivative
####################################
We can view the Richardson extrapolation step as a polynomial curve fit in the step size parameter :math:`\delta`. Our desired extrapolated value is seen as simply the constant term coefficient in that polynomial model. Remember though, this polynomial model (see :eq:`10` and :eq:`11`) has only a few terms in it with known non-zero coefficients. That is, we will expect a constant term :math:`a_0`, a term of the form :math:`a_1 \delta^4`, and a third term :math:`a_2 \delta^6`.

A neat trick to compute the statistical uncertainty in the estimate of our desired derivative is to use statistical methodology for that error estimate. While I do appreciate that there is nothing truly statistical or stochastic in this estimate, the approach still works nicely, providing a very reasonable estimate in practice. A three term Richardson-like extrapolant, then evaluated at four distinct values for :math:`\delta`, will yield an estimate of the standard error of the constant term, with one spare degree of freedom. The uncertainty is then derived by multiplying that standard error by the appropriate percentile from the Students-t distribution.

   >>> import scipy.stats as ss
   >>> allclose(ss.t.cdf(12.7062047361747, 1), 0.975)
   True

This critical level will yield a two-sided confidence interval of 95 percent.

These error estimates are also of value in a different sense. Since they are efficiently generated at all the different scales, the particular spacing which yields the minimum predicted error is chosen as the best derivative estimate. This has been shown to work consistently well. A spacing too large tends to have large errors of approximation due to the finite difference schemes used. But a too small spacing is bad also, in that we see a significant amplification of least significant fit errors in the approximation. A middle value generally seems to yield quite good results. For example, Derivative will estimate the derivative of :math:`e^x` automatically. As we see, the final overall spacing used was 0.0078125.

    >>> import numdifftools as nd
    >>> from numpy import exp, allclose
    >>> f = nd.Derivative(exp, full_output=True)
    >>> val, info = f(1)
    >>> allclose(val, 2.71828183)
    True       
    >>> allclose(info.error_estimate, 6.927791673660977e-14)
    True
    >>> allclose(info.final_step, 0.0078125)
    True


However, if we force the step size to be artificially large, then approximation error takes over.

    >>> f = nd.Derivative(exp, step=1, full_output=True) 
    >>> val, info = f(1)
    >>> allclose(val, 3.19452805)
    True
    >>> allclose(val-exp(1), 0.47624622)
    True
    >>> allclose(info.final_step, 1)
    True

And if the step size is forced to be too small, then we see noise dominate the problem.

   >>> f = nd.Derivative(exp, step=1e-10, full_output=True)
   >>> val, info = f(1)
   >>> allclose(val, 2.71828093)
   True
   >>> allclose(val - exp(1), -8.97648138e-07)
   True
   >>> allclose(info.final_step, 1.0000000e-10)
   True


Numdifftools, like Goldilocks in the fairy tale bearing her name, stays comfortably in the middle ground.


Derivative in action
####################

How does numdifftools.Derivative work in action? A simple nonlinear function with a well known derivative is :math:`e^x`. At :math:`x = 0`, the derivative should be 1.

   >>> f = nd.Derivative(exp, full_output=True)
   >>> val, info = f(0)
   >>> allclose(val, 1)
   True

   >>> allclose(info.error_estimate, 5.28466160e-14)
   True

A second simple example comes from trig functions. The first four derivatives of the sine function, evaluated at :math:`x = 0`, should be respectively :math:`[cos(0), -sin(0), -cos(0), sin(0)]`, or :math:`[1,0,-1,0]`.

    >>> from numpy import sin, allclose
    >>> import numdifftools as nd
    >>> df = nd.Derivative(sin, n=1)
    >>> allclose(df(0), 1.)
    True

    >>> ddf = nd.Derivative(sin, n=2)
    >>> allclose(ddf(0), 0.)
    True

    >>> dddf = nd.Derivative(sin, n=3)
    >>> allclose(dddf(0), -1.)
    True

    >>> ddddf = nd.Derivative(sin, n=4)
    >>> allclose(ddddf(0), 0.)
    True


Gradient and Hessian  estimation
################################

Estimation of the gradient vector (numdifftools.Gradient) of a function of multiple variables is a simple task, requiring merely repeated calls to numdifftools.Derivative. Likewise, the diagonal elements of the hessian matrix are merely pure second partial derivatives of a function. numdifftools.Hessdiag accomplishes this task, again calling numdifftools.Derivative multiple times. Efficient computation of the off-diagonal (mixed partial derivative) elements of the Hessian matrix uses a scheme much like that of numdifftools.Derivative, then Richardson extrapolation is used to improve a set of second order finite difference estimates of those mixed partials.

Multivariate calculus examples
------------------------------
Typical usage of the gradient and Hessian might be in optimization problems, where one might compare 
an analytically derived gradient for correctness, or use the Hessian matrix to compute confidence interval estimates on parameters in a maximum likelihood estimation.


Gradients and Hessians
----------------------

    >>> import numpy as np
    >>> def rosen(x): return (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    
Gradient of the Rosenbrock function at [1,1], the global minimizer
    >>> grad = nd.Gradient(rosen)([1, 1])

The gradient should be zero (within floating point noise)

    >>> allclose(grad, 0)
    True

The Hessian matrix at the minimizer should be positive definite
    >>> H = nd.Hessian(rosen)([1, 1])

The eigenvalues of H should be positive

    >>> li, U = np.linalg.eig(H)
    >>> li>0
    array([ True,  True], dtype=bool)


Gradient estimation of a function of 5 variables
    >>> f = lambda x: np.sum(x**2)
    >>> grad = nd.Gradient(f)(np.r_[1, 2, 3, 4, 5])
    >>> allclose(grad, [  2.,   4.,   6.,   8.,  10.])
    True

Simple Hessian matrix of a problem with 3 independent variables
    >>> f = lambda x: x[0] + x[1]**2 + x[2]**3
    >>> H = nd.Hessian(f)([1, 2, 3])
    >>> allclose(H, np.diag([0, 2, 18]))
    True

A semi-definite Hessian matrix
    >>> H = nd.Hessian(lambda xy: np.cos(xy[0] - xy[1]))([0, 0])

one of these eigenvalues will be zero (approximately)

    >>> np.abs(np.linalg.eig(H)[0]) < 1e-12
    array([ True, False], dtype=bool)


Directional derivatives 
-----------------------

The directional derivative will be the dot product of the gradient with the (unit normalized) vector. This is of course possible to do with numdifftools and you could do it like this for the Rosenbrock function at the solution, x0 = [1,1]:

    >>> v = np.r_[1, 2]/np.sqrt(5)
    >>> x0 = [1, 1]
    >>> directional_diff = np.dot(nd.Gradient(rosen)(x0), v)

This should be zero. 

    >>> allclose(directional_diff, 0)
    True

 
Ok, its a trivial test case, but it easy to compute the directional derivative at other locations:

    >>> v2 = np.r_[1, -1]/np.sqrt(2)
    >>> x2 = [2, 3]
    >>> directionaldiff = np.dot(nd.Gradient(rosen)(x2), v2)
    >>> allclose(directionaldiff, 743.87633380824832)
    True

Jacobian matrix
---------------
Jacobian matrix of a scalar function is just the gradient

    >>> jac = nd.Jacobian(rosen)([2, 3])
    >>> grad = nd.Gradient(rosen)([2, 3])
    >>> allclose(jac, grad)
    True

Jacobian matrix of a linear system will reduce to the design matrix

    >>> A = np.random.rand(5,3)
    >>> b = np.random.rand(5)
    >>> fun = lambda x: np.dot(x, A.T) - b
    >>> x = np.random.rand(3)
    >>> jac = nd.Jacobian(fun)(x)

This should be essentially zero at any location x

    >>> allclose(jac - A, 0)
    True

The jacobian matrix of a nonlinear transformation of variables evaluated at some
arbitrary location [-2, -3]

    >>> fun = lambda xy: np.r_[xy[0]**2, np.cos(xy[0] - xy[1])]
    >>> jac = nd.Jacobian(fun)([-2, -3])
    >>> np.allclose(jac, [[-4.,  0.],
    ...                   [-0.84147098,  0.84147098]])
    True


Conclusion
##########

numdifftools.Derivative is an a adaptive scheme that can compute the derivative of arbitrary (well behaved) functions. It is reasonably fast as an adaptive method. Many options have been provided for the user who wishes the ultimate amount of control over the estimation.


Acknowledgments
###############

The numdifftools package was originally a translation of an adaptive numerical differentiation toolbox written in Matlab by John D'Errico [DErrico2006]_.

Numdifftools has as of version 0.9 been extended with some of the functionality
found in the statsmodels.tools.numdiff module written by Josef Perktold [Perktold2014]_.


References
##########
.. [LynessMoler1966] Lyness, J. M., Moler, C. B. (1966). Vandermonde Systems and Numerical
                     Differentiation. *Numerische Mathematik*.

.. [LynessMoler1969] Lyness, J. M., Moler, C. B. (1969). Generalized Romberg Methods for
                     Integrals of Derivatives. *Numerische Mathematik*.

.. [DErrico2006] D'Errico, J. R.  (2006), Adaptive Robust Numerical Differentiation
	http://www.mathworks.com/matlabcentral/fileexchange/13490-adaptive-robust-numerical-differentiation

.. [Perktold2014] Perktold, J (2014), numdiff package   
	http://statsmodels.sourceforge.net/0.6.0/_modules/statsmodels/tools/numdiff.html

.. [LaiCrassidisCheng2005] K.-L. Lai, J.L. Crassidis, Y. Cheng, J. Kim (2005), New complex step derivative approximations with 										application to second-order kalman filtering, 
									AIAA Guidance, *Navigation and Control Conference*,
									San Francisco, California, August 2005, AIAA-2005-5944.

.. [Ridout2009] Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. *The American Statistician*, 63, 66-74

.. [NAG] *NAG Library*. NAG Fortran Library Document: D04AAF


.. rubric:: Footnotes

.. [1]
   We would normally write these additional terms using O() notation,
   where all that matters is that the error term is :math:`O(\delta)` or
   perhaps :math:`O(\delta^2)`, but explicit understanding of these
   error terms will be useful in the Richardson extrapolation step later
   on.

.. [2]
   An even function is one which expresses an even symmetry around a
   given point. An even symmetry has the property that
   :math:`f(x) = f(-x)`. Likewise, an odd function expresses an odd
   symmetry, wherein :math:`f(x) = -f(-x)`.

.. [3] For practical purposes the maximum order of the derivative is between 4 and 10
	depending on the function to differentiate and also the method used
	in the approximation.
