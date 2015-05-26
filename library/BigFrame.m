% Take the image and add a big frame
function [ImgFrame] = BigFrame(Img)
% Add a frame
for i = 1:size(Img,2)
    Img(1:5,i)=1;
    Img(end-4:end,i)=1;
end
for i = 1:size(Img,1)
    Img(i,1:5)=1;
    Img(i,end-4:end)=1;
end
ImgFrame = Img;
end