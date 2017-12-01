function C = conditioner( points, isotropic )
%Custom conditioner
%   1430003045 ÷”æ˚»Â
dim = size(points,1);

points(end + 1,:) = 1;
points = points(1:dim-1,:);

avg = mean(points,2);
s = std(points');
s = s + (s == 0);

if nargin == 1
    C = [ diag(sqrt(2)./s) - diag(sqrt(2)./s) * avg];
else
    C = [ diag(sqrt(2)./(ones(1,dim-1)*avg(s))) -diag(sqrt(2)./s) * avg];
end
C(dim,:) = 0;
C(dim,dim) = 1;
end

