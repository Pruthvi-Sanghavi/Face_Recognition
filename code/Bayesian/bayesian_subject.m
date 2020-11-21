clc
clear
tic
data = load('../datasets/data.mat');
X = reshape(data.face,504,[]);
N = size(X,2);
X_neutral = X(:,1:3:N);
X_expression = X(:,2:3:N);
X_illumination = X(:,3:3:N);

%X_train = [X_neutral X_expression];
%X_train = [X_neutral X_illumination];
X_train = [X_expression X_illumination];


%X_test = X_illumination;
X_test = X_neutral;
%X_test = X_expression;


i = 1:size(X_train,2)/2;
mu_train = (X_train(:,i) + X_train(:,i+size(X_train,2)/2))/2;



noise = 0.4;
for i = 1:size(X_train,2)/2
     cov_train(:,:,i)= (((X_train(:,i)-mu_train(:,i))*transpose((X_train(:,i)-mu_train(:,i))))+((X_train(:,i+size(X_train,2)/2)-mu_train(:,i))*transpose((X_train(:,i+size(X_train,2)/2)-mu_train(:,i)))))/2;
     cov_train(:,:,i)= cov_train(:,:,i) + noise.*eye(size(cov_train,1));
     inv_train_cov(:,:,i)=pinv(cov_train(:,:,i));  
end
accuracy = 0;
for i = 1:size(X_test,2)
    for j = 1:size(X_test,2)        
         P_test(i,j) = (1/sqrt(2*pi*det(cov_train(:,:,j))))*exp(-0.5*(X_test(:,i)-mu_train(:,j))'*inv_train_cov(:,:,j)*(X_test(:,i)-mu_train(:,j)));
         
    end
    [~,index] = max(P_test(i,:));
    if index == i

        accuracy = accuracy+1;

    end
    
end

disp("accuracy of base bayesian classifier for subject classification");
disp((accuracy/size(X_test,2))*100);
toc