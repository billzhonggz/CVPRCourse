function x = get_homg( x )
%Costum function to get non-homegenous coordinate.
%   1430003045 �Ӿ���

if isempty(x)
    return
end

x(end+1,:) = 1;

end

