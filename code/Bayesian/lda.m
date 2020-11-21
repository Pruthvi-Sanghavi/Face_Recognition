function [y_train_n, y_train_e] = lda(X_train, mu_1, mu_2)
    mu_c = (mu_1 + mu_2)/2;
    delta = 0.05;
    for i=1:size(X_train,2)/2
        
    % Within class scatter matrix
        s_w_neutral = (X_train(:,i) - mu_1)*(X_train(:,i) - mu_1)' + delta*eye(24*21);
        s_w_expression = (X_train(:,i+100) - mu_2)*(X_train(:,i+100) - mu_2)' + delta*eye(24*21);
   % Between class scatter matrix
        s_b_neutral = (mu_c - mu_1)*(mu_c - mu_1)';
        s_b_expression = (mu_c - mu_2)*(mu_c - mu_2)';
        
%     % Calculating the eigen value and eigen vectors of sigma s_w^-1*s_b
        [v_neutral, lamda_neutral] = eig(inv(s_w_neutral)*s_b_neutral);
        [v_expression, lamda_expression] = eig(inv(s_w_expression)*s_b_expression);
        
        
        l_neutral_diag = diag(lamda_neutral);
        l_expression_diag = diag(lamda_expression);
%         lamda = diag(lamda);
%         lamda = sort(lamda);
% 
        y_train_neutral(:,i) = v_neutral*X_train(:,i);
        y_train_expression(:,i) = v_expression*X_train(:,i+100);
        
    end
    y_train_n = y_train_neutral;
    y_train_e = y_train_expression;
end