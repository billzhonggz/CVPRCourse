function x = get_nunhomg( x )
%Costum function to get non-homegenous coordinate.
%   1430003045 �Ӿ���

if isempty(x)
    x = [];
    return;
end

d = size(x,1) - 1;
x = x(1:d,:)./(ones(d,1)*x(end,:));

end

