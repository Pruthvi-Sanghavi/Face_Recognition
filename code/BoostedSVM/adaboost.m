function [a,h] = adaboost(X_train, D_vect, Num_iterations, h,y)
    for i=1:size(X_train,1)
        for t=1:Num_iterations
            r = D_vect*y(505,i)*h;
            alpha = 1/2*log((1+r)/(1-r));
            H = sign(alpha*h);
        end
        
    end
    h = H;    
    a = alpha;
   
end