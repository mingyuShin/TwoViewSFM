% load two images
images = imageDatastore('data');
I1 = readimage(images, 1);
I2 = readimage(images, 2);
grayI1 = rgb2gray(I1);
grayI2 = rgb2gray(I2);
% figure
% imshowpair(I1,I2, 'montage');


kp1 = detectSIFTFeatures(grayI1);
kp2 = detectSIFTFeatures(grayI2);

[des1, pts1] = extractFeatures(grayI1, kp1); %(2581,128), (2581,1)
[des2, pts2] = extractFeatures(grayI2, kp2);

indexPairs = matchFeatures(des1, des2);

matchedI1 = pts1(indexPairs(:,1));
matchedI2 = pts2(indexPairs(:,2));

disp(matchedI1)

figure
showMatchedFeatures(I1, I2, matchedI1, matchedI2)