clear
clc
dataset = load("../datasets/data.mat"); % Loading the dataset 'data.mat' file
X = reshape(dataset.face,504,[]); % Converting dataset in the vector form X = [x1,x2,...,x600] 
N = size(X,2); % Number of samples
split = 0.5; % Split in the dataset for training and testing purpose


X_train = [X(:,1:3:N*(1-split)) X(:,2:3:N*(1-split))]; % Training images: X_train = [X_neutral X_expression]
X_test = [X(:,N*(1-split) + 1:3:N) X(:,N*(1-split) + 2:3:N)];
%X_test = X(:, N*(1-split) + 1:N); % Testing images



% Mean of neutralface and expressionface images using ML Estimation
mu_neutral = sum(X(:,1:3:N*(1-split)),2)/size(X(:,1:3:N*(1-split)),2);
mu_expression = sum(X(:,2:3:N*(1-split)),2)/size(X(:,2:3:N*(1-split)),2);



% Covariance of neutralface and expressionface images using ML Estimation
covariance_neutral = cov(X(:,1:3:N*(1-split))');
covariance_expression = cov(X(:,2:3:N*(1-split))');

% Get the determinant
I = eye(size(covariance_neutral));
noise = 0.4*I;
covariance_neutral = covariance_neutral + noise;
covariance_expression = covariance_expression + noise;

% Inverse of Covariance matrix
% Finding the pseudo inverse matrix since the matrix is singular
cov_neut_inv = pinv(covariance_neutral);
cov_expression_inv = pinv(covariance_expression);

accuracy = 0;
for n = 1:size(X_test,2)
    if n <= size(X_test,2)/2
        true_label = 1;
    else
        true_label = -1;
    end
    %creating model for class_neutral and class_expression
    P_neutral = (1/sqrt(2*pi*det(covariance_neutral)))*exp(-0.5*(X_test(:,n)-mu_neutral)'*cov_neut_inv*(X_test(:,n)-mu_neutral));
    P_expression = (1/sqrt(2*pi*det(covariance_expression)))*exp(-0.5*(X_test(:,n)-mu_expression)'*cov_expression_inv*(X_test(:,n)-mu_expression));
    
    posteriors = [P_neutral 1;P_expression -1];
    %finding max of the two posterior probabilities
    [~,index] = max(posteriors(:,1));
    
    %proper labelling for comparison
    if index == 1
        computed_label = 1;
    elseif index == 2
        computed_label = -1;
    end
    
    %comparison of labels
    if true_label*computed_label == 1
        accuracy = accuracy+1;
    end
end
disp('Acccuracy of the bayesian classifier: ');
disp((accuracy/size(X_test,2))*100);

%%%%%%%%%%%%%%%%%%%%%%%%% Plotting the results %%%%%%%%%%%%%%
split_vector = [0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90];
accuracy_vector = [83.75, 83.33, 86.875, 87.00, 88.33, 86.7857, 84.0625, 84.1667];


xlim([0.20,0.90]);
ylim([0,100]);
title('Split ratio v.s. Accuracy')
xlabel("Split Ratio")
ylabel('Accuracy')
%legend({'base'},'Location','southwest')
grid
hold on;
plot(split_vector(1,:), accuracy_vector(1,:));



