% Take the image and retrn the Corp coordinate
function [xmin,xmax,ymin,ymax] = Corpinfo(ImgHand)
[numx, numy] = size(ImgHand);
Image = ~ImgHand;
% for x
for i = 1:numx
    check = sum(Image(i,:));
    if check ~=0
        xmin = i-5;
        break
    end
end
for i = numx:-1:1
    check = sum(Image(i,:));
    if check ~=0
        xmax = i+5;
        break
    end
end
% for y    
for i = 1:numy
    check = sum(Image(:,i));
    if check ~=0
        ymin = i-5;
        break
    end
end
for i = numy:-1:1
    check = sum(Image(:,i));
    if check ~=0
        ymax = i+5;
        break
    end
end
end