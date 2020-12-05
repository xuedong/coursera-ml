function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = - dot(y, log(sigmoid(X*theta)))/m - dot(1-y, log(1-sigmoid(X*theta)))/m + dot(theta, theta)*lambda/(2*m);
J(1) = J(1) - theta(1)*theta(1)*lambda/(2*m);
grad = X'*(sigmoid(X*theta)-y)/m + theta*lambda/m;
grad(1) = grad(1) - theta(1)*lambda/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
