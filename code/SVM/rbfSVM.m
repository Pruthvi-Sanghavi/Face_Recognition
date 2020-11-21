clear
clc
data = load('../datasets/data.mat');
X = reshape(data.face,24*21,[]);
split = 1/2; % Training data = split*X, Testing data = (1 - split)*X
N = size(X,2);
sigma = 15.5;
acc = 0;

X_neutral_train = X(:,1:3:split*N);
X_expression_train = X(:,2:3:split*N);
X_train = [X_neutral_train X_expression_train]';

X_neutral_test = X(:,split*N + 1:3:N);
X_expression_test = X(:,split*N + 2:3:N);
X_test = [X_neutral_test X_expression_test];

% Assigning labels to the training data
% 1 for neutral face and -1 for expression face
C = 0.25;

labels = [ones(size(X_neutral_train,2),1); -ones(size(X_expression_train,2),1)];
X_train_labelled = [X_train labels];

% creating labelled Gram Matrix
%H = (X_train*X_train').*(labels*labels');
%%%%%%%%%%%%%%%%%%%% KERNEL RBF %%%%%%%%%%%%%%%%%%%%%%
K = [];
for n = 1:size(X_train,1)
    for m = 1:size(X_train,1)
        K(n,m) = exp(-((norm(X_train(:,n)-X_train(:,m)))^2)/sigma^2);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
mu = quadprog(K,f,[],[],B,Beq,lb,ub);

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

%%%%%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%
K_test = [];
for n = 1:size(X_test,2)
    for m = 1:size(X_test,2)

        K_test(n,m) = exp(-((norm(X_test(:,n)-X_test(:,m)))^2)/sigma^2);

    end
end
for i = 1:size(K_test,2)
        % asssigning true labels: +1 for upto 1st half of the testing_set
        % and -1 for the 2nd half of the testing_set
        if i <= size(X_train,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        % using the test image in the linear predictor
        value = theta'.*K_test(:,i) + theta0;
        % multiplying value with the true_label for easy comparison to
        % evaluate accuracy
        prediction = value*true_label;
        
        % self-explainatory
        if prediction > 0
            acc = acc+1;
        end
        
end
accuracy = (acc/size(K_test,2))*100;
%accuracy = kernelTesting('rbf',sigma,theta,theta0,X_test);
%accuracy = testing(theta,theta0,X_test);
disp('base accuracy:');
disp(accuracy);

sigma = [0 5 10 15 20 25 30];
accuracy_rbf = [0 0 0 0 100 100 100];
xlim([0,30]);
ylim([0,100]);
title('sigma v.s. accuracy (split = 3/4)')
xlabel("sigma")
ylabel('Accuracy')
%legend({'base'},'Location','southwest')
grid
hold on;
plot(sigma(:),accuracy_rbf(:));
