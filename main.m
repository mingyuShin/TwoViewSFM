%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the skeleton code of PA1 in IC614 Computer Vision.               %
% It will help you to implement the Structure-from-Motion method easily.   %
% Using this skeleton is recommended, but it's not necessary.              %
% You can freely modify it or you can implement your own program.          %
% If you have a question, please send me an email to sunghoonim@dgist.ac.kr%
%                                                      Prof. Sunghoon Im   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  202022010 김진회 two-view initialization
%   non-Marked --> self implementation
%       Marked --> Copied or Refered
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for-practice I made it for left-coordinate system to compare with
% internal matlab function(they use it....)
% may be there are some confusions...


close all;
clear all;
run('vlfeat-0.9.21/toolbox/vl_setup')
addpath('Givenfunctions');
addpath('vlfeat-0.9.21');
%% read images and define camera intrinsics
images = imageDatastore('data');

% given datasets
% img1 = readimage(images, 1);
% img2 = readimage(images, 2);
% 
% filename = sprintf('%02dviews.ply', 2);

% own datasets
img1 = readimage(images, 3);
img2 = readimage(images, 4);

filename = sprintf('%02dviewsOwn.ply', 2);

% camera intrinsics for essential matrix decomposition and triangulation
% given datasets
% temp      =       [ 1698.873755   0.000000       971.7497705;
%                     0.000000      1698.8796645   647.7488275;
%                     0.000000      0.000000       1.000000 ];
% 
% ThresholdK = 0.001;
% iterK = 6000;


% own dataset IntrinsicMatrix
% temp = [2947.63139413012,  0,                   2009.35816825727;
%          0,                 2955.26831604551,    1479.60847982218;
%          0,                 0,                   1];
% 
% ThresholdK = 0.0001;
% iterK = 8000;

% own dataset2

% own datasets
img1 = readimage(images, 5);
img2 = readimage(images, 6);

filename = sprintf('%02dviewsOwn2.ply', 2);

temp = [ 3164.63218 0.000000     1994.02346;
        0.000000    3166.59599   1452.02944;
        0.000000    0.000000     1.000000 ];
ThresholdK = 0.0001;
iterK = 4000;

% SIFT feature extraction
[kp1, kp2] = SIFTFeatureExtract(img1, img2);

homo_kp1 = vertcat(kp1, ones([length(kp1),1])'); % (3,125)
homo_kp2 = vertcat(kp2, ones([length(kp2),1])'); % (3,125)

% multiple intrinsic matrix
homo_normal_kp1 = temp\homo_kp1; %(3,5)
homo_normal_kp2 = temp\homo_kp2; %(3,5)

Max = 0;

%% select five key points with RANSAC
% choose best essential matrix

for i=1:iterK
   idx = randperm(length(kp1), 5);
   selectKp1 = homo_normal_kp1(:, idx); %(3,5)
   selectKp2 = homo_normal_kp2(:, idx); %(3,5)

%     Given Five Point Algorithm
    Evec = calibrated_fivepoint(selectKp1, selectKp2);
   for j=1:size(Evec,2)
       eMatrix = reshape(Evec(:,j), 3, 3);
       numerator = abs(diag(homo_normal_kp1'*eMatrix*homo_normal_kp2)); %kp2 = x, kp1 = x'
       temp_denom = eMatrix * homo_normal_kp2;
       temp_denom(3,:) = [];
       denom = vecnorm(temp_denom, 2, 1);
       Edistance = numerator ./ denom';
       inliner = Edistance < ThresholdK;
       if sum(inliner) > Max
           Max = sum(inliner);
           inliner_idx = find(inliner);
           MaxEMat = eMatrix;
       end
   end
end


color = [];
for i=1:length(inliner_idx)
    tcolor = img2(round(kp2(2, inliner_idx(i))), round(kp2(1, inliner_idx(i))), :);
    tcolor = reshape(tcolor, 3, 1);
    color = [color, tcolor];
end
color = double(color);
%% draw Epipolar Lines
Fund = inv(temp)'*MaxEMat*inv(temp);

figure;
subplot(121);
imshow(img1); 
title('Inliers and Epipolar Lines in First Image'); hold on;
plot(kp1(1, inliner_idx), kp1(2,inliner_idx),'go')

epiLines = epipolarLine(Fund,kp2(:, inliner_idx)');
points = lineToBorderPoints(epiLines,size(img1));
line(points(:,[1,3])',points(:,[2,4])');

subplot(122); 
imshow(img2);
title('Inliers and Epipolar Lines in Second Image'); hold on;
plot(kp2(1,inliner_idx), kp2(2, inliner_idx),'go')

epiLines = epipolarLine(Fund',kp1(:, inliner_idx)');
points = lineToBorderPoints(epiLines,size(img2));
line(points(:,[1,3])',points(:,[2,4])');
truesize;
saveas(gcf,'EpipolarLinesOwn2.png')

%% Triangulation with SVD
[U,D,V] = svd(MaxEMat);
diag_11 = [1,0,0; 0,1,0; 0,0,0];
E = U*diag_11*V';
[U,D,V] = svd(E);

W = [0,-1,0; 1,0,0; 0,0,1];
T = U(:,3);
R1 = U*W*V';
R2 = U*W'*V';

P1 = [R1, T];
P2 = [R1, -T];
P3 = [R2, T];
P4 = [R2, -T];

P = {P1, P2, P3, P4};
P0 = eye(3,4);

for i=1:length(inliner_idx)
    A = [homo_normal_kp2(1,inliner_idx(i))*P0(3,:)-P0(1,:); homo_normal_kp2(2,inliner_idx(i))*P0(3,:)-P0(2,:);homo_normal_kp1(1,inliner_idx(i))*P1(3,:)-P1(1,:);homo_normal_kp1(2,inliner_idx(i))*P1(3,:)-P1(2,:)];
    [~,~,V1] = svd(A);
    Xtemp(:,i) = V1(:,4);
end
Xtemp = Xtemp ./ Xtemp(4,:);
X14 = Xtemp(1:3,:);

for i=1:length(inliner_idx)
    A = [homo_normal_kp2(1,inliner_idx(i))*P0(3,:)-P0(1,:); homo_normal_kp2(2,inliner_idx(i))*P0(3,:)-P0(2,:); homo_normal_kp1(1,inliner_idx(i))*P2(3,:)-P2(1,:); homo_normal_kp1(2,inliner_idx(i))*P2(3,:)-P2(2,:)];
    [~,~,V1] = svd(A);
    Xtemp(:,i) = V1(:,4);
end
Xtemp = Xtemp ./ Xtemp(4,:);
X24 = Xtemp(1:3,:);

for i=1:length(inliner_idx)
    A = [homo_normal_kp2(1,inliner_idx(i))*P0(3,:)-P0(1,:); homo_normal_kp2(2,inliner_idx(i))*P0(3,:)-P0(2,:);homo_normal_kp1(1,inliner_idx(i))*P3(3,:)-P3(1,:);homo_normal_kp1(2,inliner_idx(i))*P3(3,:)-P3(2,:)];
    [~,~,V1] = svd(A);
    Xtemp(:,i) = V1(:,4);
end
Xtemp = Xtemp ./ Xtemp(4,:);
X34 = Xtemp(1:3,:);

for i=1:length(inliner_idx)
    A = [homo_normal_kp2(1,inliner_idx(i))*P0(3,:)-P0(1,:); homo_normal_kp2(2,inliner_idx(i))*P0(3,:)-P0(2,:);homo_normal_kp1(1,inliner_idx(i))*P4(3,:)-P4(1,:);homo_normal_kp1(2,inliner_idx(i))*P4(3,:)-P4(2,:)];
    [~,~,V1] = svd(A);
    Xtemp(:,i) = V1(:,4);
end
Xtemp = Xtemp ./ Xtemp(4,:);
X44 = Xtemp(1:3,:);

X = {X14, X24, X34, X44};



depth = [];
for i=1:4
    RP = P{1,i};
    RX = X{1,i};

    M2 = P0(:,1:3);
    w2 = M2(3,:)*RX;

    M1 = RP(:,1:3);
    w1 = M1(3,:)*(RX-RP(:,4));
    depth = [depth, sum(sign(w2))+sum(sign(w1))];
end

[~,I] = max(depth);

RP = P{1,I};
RX = X{1,I};

X_exist = vertcat(RX, color);
X_exists = X_exist(:,find(RX(3,:)>=0));
X_exists = horzcat(X_exists, [0,0,0,255,0,0]');
X_exists = horzcat(X_exists,[-RP(:,4)',255,0,0]');
SavePLY(filename, X_exists)

