function x = get_homg( x )
%Costum function to get non-homegenous coordinate.
%   1430003045 ÖÓ¾ûÈå

if isempty(x)
    return
end

x(end+1,:) = 1;

end

