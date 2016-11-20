function cost_i = objective(d_i, I_i, I_j)
    omega = 1:41;
    g = gamma(omega);
    cost_i = norm(I_i * exp(d_i*g - I_j));
end