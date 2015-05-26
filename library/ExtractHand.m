% Extract hand information based on Boundary Edge
function [wristpoint, angle, Orientation] = ExtractHand(FimgBoundaryEdge)
imgsize = size(FimgBoundaryEdge);
% Chenck the orientation of the hand
Up = sum(FimgBoundaryEdge(2,:));
Left = sum(FimgBoundaryEdge(:,2));
Right = sum(FimgBoundaryEdge(:,end-1));
Orientation = 0;
if Up >= 2
    Orientation = 180;
    FimgBoundaryEdge = imrotate(FimgBoundaryEdge,Orientation);
elseif Left >= 2
    Orientation = 90;
    FimgBoundaryEdge = imrotate(FimgBoundaryEdge,Orientation);
elseif Right >= 2
    Orientation = 270;
    FimgBoundaryEdge = imrotate(FimgBoundaryEdge,Orientation);
end
upbound = round(2*imgsize(1)/3);
contactpoints = find(FimgBoundaryEdge(end-1,:)==1,10);
contactpoints = [min(contactpoints),max(contactpoints)];
Elbow = FimgBoundaryEdge(upbound:end,:);
[wristpoint, angle]  = Wrist(Elbow, contactpoints, upbound);
end