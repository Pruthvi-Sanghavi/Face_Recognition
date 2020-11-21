function [X_pca] = pca(X)

    % Calculating the eigen values and eigen vectors
    [W,S,V] = svds(X,200);
    % Transforming the space
    X_pca = W'*X;

end