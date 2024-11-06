= DSGE model

$E [f(y_(t-1), y_t, y_(t+1), epsilon)] = 0$

- f a set of equations
  - euler equations
  - market clearing
  - accounting

- perturbations:
  - steady-state
  - derivatives
  - blanchard-kahn conditions
  - higher order expansions


- deeplearning: 
  - solution: $y_t=phi(y_(t-1), epsilon_t) $
  - approximate: $y_t=N(y_(t-1), epsilon_t \; theta) $


= Global Method

- python
  - arrays
  - automatic differentiation
  

- modeling:
  - agent:

    - explicit optimization problem: $max_c E sum beta^t U(c_t)$
    - transition equations: $w_t = (w_(t-1)-c_(t-1))r + y_t$
  - market structure