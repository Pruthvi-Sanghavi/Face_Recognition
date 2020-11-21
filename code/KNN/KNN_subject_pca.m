clear
clc

l = 24*21; % Size of the image
dataset = load('../datasets/data.mat'); % Load the dataset
X = reshape(dataset.face,l,[]); % Dataset in the form X = [x_1, x_2,...x_600]
N = size(X,2); % Number of Samples
K = 8;

[X_pca, w, v, s] = pca(X);

X_neutral = X_pca(:,1:3:N);
X_expression = X_pca(:,2:3:N);
X_illumination = X_pca(:,3:3:N);

%X_train = [X_neutral X_expression];
%X_train = [X_neutral X_illumination];
X_train = [X_expression X_illumination];

X_test = X_neutral;
%X_test = X_expression;
%X_test = X_illumination;


accuracy = 0;
for n = 1: size(X_test, 2)
    distance_vector = [];
    if n <= size(X_test, 2)
        true_label = 1;
    else
        true_label = -1;
   
    end
    
    %%%%%%%%%%%% Neutral - Expression %%%%%%%%%%%%%%%%%%
%     for m = 1: size(X_neutral, 2)
%         %distance = L2_norm(testing_set(:,n), class_neutral(:,m));
%         distance = norm(X_test(:,n)-X_neutral(:,m));
%         distance_vector = [distance_vector;[distance 1]];
%     end
%     
%     
%     for m = 1: size(X_expression, 2)
%         %distance = L2_norm(testing_set(:,n), class_expression(:,m));
%         distance = norm(X_test(:,n)-X_expression(:,m));
%         distance_vector = [distance_vector;[distance -1]];
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%% Neutral - Illumination %%%%%%%%%%%%%%%
%     for m = 1: size(X_neutral, 2)
%         %distance = L2_norm(testing_set(:,n), class_neutral(:,m));
%         distance = norm(X_test(:,n)-X_neutral(:,m));
%         distance_vector = [distance_vector;[distance 1]];
%     end
%     
%     
%     for m = 1: size(X_illumination, 2)
%         %distance = L2_norm(testing_set(:,n), class_expression(:,m));
%         distance = norm(X_test(:,n)-X_illumination(:,m));
%         distance_vector = [distance_vector;[distance -1]];
%     end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%% Expression - Illumination %%%%%%%%%%%%%%%
    for m = 1: size(X_expression, 2)
        %distance = L2_norm(testing_set(:,n), class_neutral(:,m));
        distance = norm(X_test(:,n)-X_expression(:,m));
        distance_vector = [distance_vector;[distance 1]];
    end
    
    
    for m = 1: size(X_illumination, 2)
        %distance = L2_norm(testing_set(:,n), class_expression(:,m));
        distance = norm(X_test(:,n)-X_illumination(:,m));
        distance_vector = [distance_vector;[distance -1]];
    end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



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