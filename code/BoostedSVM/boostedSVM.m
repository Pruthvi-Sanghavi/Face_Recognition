clear
clc
data = load('../datasets/data.mat');
X = reshape(data.face,24*21,[]);
split = 1/2; % Training data = split*X, Testing data = (1 - split)*X
N = size(X,2);
T = 1000;
D = 1/N;

X_neutral_train = X(:,1:3:split*N);
X_expression_train = X(:,2:3:split*N);
X_train = [X_neutral_train X_expression_train]';



X_neutral_test = X(:,split*N + 1:3:N);
X_expression_test = X(:,split*N + 2:3:N);
X_test = [X_neutral_test X_expression_test];

% Assigning labels to the training data
% 1 for neutral face and -1 for expression face
C = 0.4;

labels = [ones(size(X_neutral_train,2),1); -ones(size(X_expression_train,2),1)];
X_train_labelled = [X_train labels]';

% creating labelled Gram Matrix
H = (X_train*X_train').*(labels*labels');

% modelling parameters according to the ones given in the documentation of
% QuadProg.
f = -ones(size(X_train,1),1);
B = [labels';zeros(size(X_train,1)-1,size(X_train,1))];
Beq = zeros(size(X_train,1),1);

% additional condition for non-separable classes here. The lower bound is
% 0, and the upperbound is C
lb = zeros(size(X_train,1),1);
ub = C*ones(size(lb));

% solving the minimization problem using the 'quadprog' function in MATLAB.
mu = quadprog(H,f,[],[],B,Beq,lb,ub);

mu_ = [];
for i = 1:size(mu,1)
    if mu(i) <= 10^-8
        mu_(i) = 0;
    else
        mu_(i) = mu(i);
    end
end
% mu_ is an appropriate vector with small values reduced to zeros.
mu_ = mu_';

% obtaining the values of wt. vector and bias term for linear
% classification
theta = ((mu_.*labels)'*X_train)';

% looking for random index of non-zero mu value (support-vector) to use it
% to find the bias (theta0)
[~,index] = max(mu_);
theta0 = (1/labels(index)) - theta'*X_train(index,:)';

[accuracy, predict, vals, tl, acu] = testing(theta,theta0,X_test);
[a, h] = adaboost(X_train, D, T, tl, X_train_labelled);



disp('base accuracy:');
disp(accuracy);


