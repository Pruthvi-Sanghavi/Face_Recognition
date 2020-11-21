function [theta,theta0] = Kernelsvm('poly',r,Ck,class_neutral,class_expression)
    acc = 0;
    % kernel transformation starts here: according to the name of the
    % kernel and the parameter for the given kernel. (sigmaSq or r)
    K = [];
    for n = 1:size(testing_set,2)
        for m = 1:size(testing_set,2)
            if strcmp(kernel,'rbf')
                % radial basis function kernel
                K(n,m) = exp(-((norm(testing_set(:,n)-testing_set(:,m)))^2)/(param));
            elseif strcmp(kernel,'poly')
                % ploynomial kernel
                K(n,m) = (testing_set(:,n)'*testing_set(:,m)+1)^param;
            end
        end
    end

end