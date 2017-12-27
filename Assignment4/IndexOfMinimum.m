function minIndex=IndexOfMinimum(x)
% This algorithm works only on one dimensional array (column vector)
% input:
% x-input array-column vector
% MinIndex = Scalar containing the index of the first minimum element. If input vector has multiple minimum values, the index of only the first is returned

minIndex=[];%Matrix to be returned in case the input vector is not a column vector or the input vector is an empty vector

if size(x,2)>1
    fprintf('Give a column vector as an input\n')
else
min_element=min(x);

for i=1:size(x,1)
    if x(i,1) == min_element
        minIndex=i;
        break;
    end
end
end

end

        
