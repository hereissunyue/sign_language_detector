% Take an Gray Scale Elbow and upbound of the image return the wristpoint  and hand angle
function [wristpoint, angle] = Wrist(Elbow, contactpoints, upbound)
[Leftlist, Rightlist] = BinaryBoundary(Elbow, contactpoints); % The list of 1 points on the frame
numleft = size(Leftlist,1);
numright = size(Rightlist,1);
Checking = [];
for i = 1:numleft
    temp = Rightlist - repmat(Leftlist(i,:),numright,1);
    [Minimum,Index] = min(sqrt(sum(transpose(temp.^2))));        
    Checking = [Checking; Index, Minimum]; %#ok<AGROW>
end
[~,I]=min(Checking(:,2));
wristpoint = [Leftlist(I,:);Rightlist(Checking(I,1),:)];
wristpoint(1,1) = wristpoint(1,1) + upbound;
wristpoint(2,1) = wristpoint(2,1) + upbound;
temp = wristpoint(1,:)-wristpoint(2,:);
angle = radtodeg(atan(temp(1)/temp(2)));
end