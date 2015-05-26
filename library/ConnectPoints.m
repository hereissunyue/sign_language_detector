% Take the wristpoint in and return the connect list
function [Connectlist] = ConnectPoints(wristpoint)
k = (wristpoint(2,1)-wristpoint(1,1))/(wristpoint(2,2)-wristpoint(1,2));
d = wristpoint(1,1) - k*wristpoint(1,2);
Connectlist = [];
for i = wristpoint(1,2)-1:wristpoint(2,2)+1
    Connectlist = [Connectlist;round(k*i+d),i;round(k*i+d)+1,i;round(k*i+d)-1,i]; %#ok<AGROW>
end
end