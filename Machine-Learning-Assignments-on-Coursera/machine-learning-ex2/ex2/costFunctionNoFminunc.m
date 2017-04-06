function [theta_new, J] = costFunctionNoFminunc(initial_theta, X, y)
%COSTFUNCTIONNOFMINUNC Compute theta and cost for logistic regression
%   [theta_new, cost] = COSTFUNCTION(theta, X, y, MaxIter) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
theta_old = initial_theta;
theta_new = zeros(size(initial_theta'));
grad = zeros(size(initial_theta));
alpha = 0.0008;
% You need to return the following variables correctly 
J = 0;
theta_new = zeros(size(initial_theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
while 1
    h = sigmoid(X * theta_old);
    J = - ((y' * log(h)) + (1 - y)' * log(1 - h)) / m;
    grad = X' * (h - y) / m;
    theta_new = theta_old - alpha * grad;
    theta_old = theta_new;
    if(J < 0.2)
       break; 
    end
end

% =============================================================

end
