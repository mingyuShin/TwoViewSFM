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
matcher = [];

% min1, min2 distance 구하기
for i = 1:length(des1)
    sub = des2-des1(i,:); %2454 128
    temp_matcher = [];
    for j = 1:length(des2)
        temp_matcher = [temp_matcher; norm(sub(j,:))];
    end
    [first_min_value, index1] = min(temp_matcher);
    A1 = temp_matcher;
    A1(index1) = [];
    second_min_value = min(A1);
    matcher = [matcher; i first_min_value, second_min_value, index1];
end

% distance 차이가 적은 것 삭제
c = pts1(matcher(1,1));
d = pts2(matcher(1,4));
for i = 2:length(matcher)
    temp = matcher(i,:);
    if temp(2) < 0.6*temp(3)
        c = vertcat(c, pts1(temp(1)));
        d = vertcat(d, pts2(temp(4)));
    end
end

disp(c)
disp(d)
figure(1)
imshow(I1)
hold on
plot(c)

figure(2)
imshow(I2)
hold on
plot(d)

figure(3)
showMatchedFeatures(I1, I2, c, d);
