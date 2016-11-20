% function cost_i = objective(d, g, I_i, I_j)
%     M = size(d,1);
%     I_nc_tilde = M * I_i ./ sum(exp(d*g'));
%     cost_i = sum((I_nc_tilde - I_j).^2);
% end

function cost_i = objective(d, g, I_i, I_j)
    cost_i = sum((I_i - d*I_j).^2);
end