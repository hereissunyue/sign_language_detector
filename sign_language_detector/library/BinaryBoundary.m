% Take an Gray Scale image and the contactpoints return the two points list
function [Leftlist, Rightlist] = BinaryBoundary(Elbow, contactpoints)
List = PointLocation(Elbow); % The position of pixel valued 1
center = round(mean(contactpoints)); % the center of the elbow
Leftlist = [];
Rightlist = [];
for i = 1: size(List,1)
    if List(i,2) > center
        Rightlist = [Rightlist; List(i,:)]; %#ok<AGROW>
    elseif List(i,2) < center
        Leftlist = [Leftlist; List(i,:)]; %#ok<AGROW>
    end
end
end