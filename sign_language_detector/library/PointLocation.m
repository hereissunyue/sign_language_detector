% Take an Gray Scale image and return location of point 1
function [list] = PointLocation(image)
list = [];
for i = 1: size(image,1)
    for j = 1:size(image,2)
        if image(i,j)==1
            list = [list;i,j]; %#ok<AGROW>
        end
    end
end
end