function K = kernel_rbf(X1,X2)
    global gamma
    K = zeros(size(X1,1),size(X2,1));
    for i=1:size(X1,1)
        for j=1:size(X2,1)
            K(i,j) = exp(-gamma*norm(x_i - x_j)^2);
        end
    end
end