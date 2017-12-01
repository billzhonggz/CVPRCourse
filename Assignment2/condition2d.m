function pc = condition2d( p,C )
%Custom 2D condition function.
%   1430003045 ÷”æ˚»Â
[r,c] = size(p)
if r == 2
    pc = get_nunhomg(C * get_homg(p));
else if r == 3
        pc = C * p
    else
        error('rows != 2 or 3');
    end
end
end
