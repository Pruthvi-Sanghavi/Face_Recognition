clear
clc
data = load('../datasets/data.mat');
X = reshape(data.face,24*21,[]);
split = 1/2; % Training data = split*X, Testing data = (1 - split)*X
N = size(X,2);
r = 1;
acc = 0;
X_neutral_train = X(:,1:3:split*N);
X_expression_train = X(:,2:3:split*N);
X_train = [X_neutral_train X_expression_train]';

X_neutral_test = X(:,split*N + 1:3:N);
X_expression_test = X(:,split*N + 2:3:N);
X_test = [X_neutral_test X_expression_test];


C = 0.40;

labels = [ones(size(X_neutral_train,2),1); -ones(size(X_expression_train,2),1)];
X_train_labelled = [X_train labels];
accu = 0;

%%%%%%%%%%%%%%%%%%%% KERNEL RBF Training %%%%%%%%%%%%%%%%%%%%%%
K = [];
for n = 1:size(X_train,1)
    for m = 1:size(X_train,1)
        K(n,m) = (X_train(:,n)'*X_train_labelled(:,505)+1)^r;
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

accuracy = testing(theta, theta0, X_test);

%%%%%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%
% K_test = [];
% X_test_t = X_test';
% for n = 1:size(X_test,2)
%     for m = 1:size(X_test,2)
% 
%         K_test(n,m) = (X_test_t(:,n)'*X_test_t(:,m)+1).^r;
% 
%     end
% end
% for i = 1:size(K_test,2)
%         % asssigning true labels: +1 for upto 1st half of the testing_set
%         % and -1 for the 2nd half of the testing_set
%         if i <= size(X_train,2)/2
%             true_label = 1;
%         else
%             true_label = -1;
%         end
%         
%         % using the test image in the linear predictor
%         value = theta'.*K_test(:,i) + theta0;
%         % multiplying value with the true_label for easy comparison to
%         % evaluate accuracy
%         prediction = value*true_label;
%         
%         % self-explainatory
%         if prediction > 0
%             accu = accu+1;
%         end
%         
% end
% accurac= (acc/size(X_test,2))*100;

disp('base accuracy:');
disp(accuracy);













