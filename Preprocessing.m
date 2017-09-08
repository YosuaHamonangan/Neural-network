function in = Preprocessing(input)
[a,b] = size(input);
maxI = max(input);
minI = min(input);

for B=1:b;
    in(:,B) = (2*(input(:,B)-minI(B))/(maxI(B)-minI(B)))-1;
end

end