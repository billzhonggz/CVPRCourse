function H = DLT( xs1,xs2 )
%DLT Implementation
%   Detailed explanation goes here

[r,c] = size(xs1);

if (size(xs1,1) == 2)
    xs1 = [xs1 ; ones(1,size(xs1,2))];
    xs2 = [xs2 ; ones(1,size(xs2,2))];
end

C1 = conditioner(xs1);
C2 = conditioner(xs2);
xs1 = condition2d(xs1,C1);
xs2 = condition2d(xs2,C2);

D = [];
z = zeros(1,3);
for k = 1:c
    p1 = xs1(:,k);
    p2 = xs2(:,k);
    D = [ D;
        p1'* p2(3) z -p1'*p2(1)
        z p1'*p2(3) -p1'*p2(2)
        ];
end

% Nullspace
[u,s,v] = svd(D,0);
s = diag(s);
% Dimension of the nullspace
ndim = sum(s < eps * s(1) * 1e3);
if ndim > 1
    fprintf('Nullspace error.');
end

h = v(:,9)
H = reshape(h,3,3)';

H = inv(C2) * H * C1;
H = H / H(3,3);
end

