%PointCloudMag


mags = 8;

fileID = fopen('PracticeData/MagDat5.txt','r');
formatSpec = '%f';
sizeA = [3 Inf];

A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

[xs,num] = size(A);
num = num/mags;

B = reshape(A,3,mags,num);
B = permute(B,[2 1 3]);

arr = ['r','g','b','k','y','m','c','w','r'];

% 3 has broken z
% 4 has huge Z offset

OverallCent = [];
RadiiAll = [];

for c=1:8
    
x1=B(c,1,:);
y1=B(c,2,:);
z1=B(c,3,:);


Ell = [x1; y1; z1];
Ell = permute(Ell,[1,3,2]);
Ell = Ell';

scatter3(x1,y1,z1,'MarkerFaceColor',arr(c))
hold on

[ center, radii, evecs, v, chi2 ] = ellipsoid_fit(Ell,'0')

[x,y,z]=ellipsoid(center(1),center(2),center(3),radii(2),radii(1),radii(3));
surf(x,y,z,'FaceAlpha',.1,'FaceColor',arr(c),'EdgeAlpha',.5)




x1= x1-center(1);
y1= y1-center(2);
z1= z1-center(3);

x1= x1/radii(3)*400;
y1= y1/radii(2)*400;
z1= z1/radii(1)*400;


scatter3(x1,y1,z1,'MarkerFaceColor',arr(c+1))
hold on

Ell = [x1; y1; z1];
Ell = permute(Ell,[1,3,2]);
Ell = Ell';

[ center, radii, evecs, v, chi2 ] = ellipsoid_fit(Ell,'0');

[x,y,z]=ellipsoid(center(1),center(2),center(3),radii(2),radii(1),radii(3));
surf(x,y,z,'FaceAlpha',.1,'FaceColor',arr(c+1),'EdgeAlpha',.5)



OverallCent = [OverallCent center];
RadiiAll = [RadiiAll radii]; 

pause

clf
 

end


axis equal
hold off


%Practice Github Comment