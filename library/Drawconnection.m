% Take the image erosion and draw the connection
function [ImgErconnect] = Drawconnection(ImgEr, Connectlist)
for i = 1 : size(Connectlist,1)
    ImgEr(Connectlist(i,1),Connectlist(i,2))=1;
end
for i = 1:size(ImgEr,2)
    ImgEr(1,i)=1;
    ImgEr(end,i)=1;
end
% Add a frame
for i = 1:size(ImgEr,1)
    ImgEr(i,1)=1;
    ImgEr(i,end)=1;
end
ImgErconnect = ImgEr;
end