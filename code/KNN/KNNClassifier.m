clear
clc

l = 24*21; % Size of the image
dataset = load('../datasets/data.mat'); % Load the dataset
X = reshape(dataset.face,l,[]); % Dataset in the form X = [x_1, x_2,...x_600]
N = size(X,2); % Number of Samples
K = 4;
split = 1/2;

X_neutral_train = X(:,1:3:split*N);
X_expression_train = X(:,2:3:split*N);
X_train = [X_neutral_train X_expression_train];

X_neutral_test = X(:,split*N + 1:3:N);
X_expression_test = X(:,split*N + 2:3:N);
X_test = [X_neutral_test X_expression_test];

accuracy = 0;
for n = 1: size(X_test, 2)
    distance_vector = [];
    if n <= size(X_test, 2)/2
        true_label = 1;
    else
        true_label = -1;
    end
    %computing L2norm or a testing image to all images in class_neutral
    %with appending label = +1.
    for m = 1: size(X_neutral_train, 2)
        %distance = L2_norm(testing_set(:,n), class_neutral(:,m));
        distance = norm(X_test(:,n)-X_neutral_train(:,m));
        distance_vector = [distance_vector;[distance 1]];
    end
    %computing L2norm or a testing image to all images in class_expression
    %with appending label = -1.
    for m = 1: size(X_expression_train, 2)
        %distance = L2_norm(testing_set(:,n), class_expression(:,m));
        distance = norm(X_test(:,n)-X_expression_train(:,m));
        distance_vector = [distance_vector;[distance -1]];
    end
    %find the computed label using the value of K from distance_vector
    %predicted_label = predictknn(K,distance_vector);
    [B,i] = mink(distance_vector(:,1),K);
    for m=i(:)
        index = distance_vector(m,2);
        if mode(index) == 1
            prediction = 1;
        elseif mode(index) == -1
            prediction = -1;
        end
    end
    if true_label*prediction == 1
         accuracy = accuracy + 1;
    end
end
disp('Base acccuracy: ');
disp(accuracy/size(X_test, 2)*100);




    