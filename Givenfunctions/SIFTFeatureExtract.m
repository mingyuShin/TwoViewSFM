function [matchedI1, matchedI2] = SIFTFeatureExtract(img1, img2)
    grayI1 = rgb2gray(img1);
    grayI2 = rgb2gray(img2);
    

    [pts1, des1] = vl_sift(single(grayI1));
    [pts2, des2] = vl_sift(single(grayI2));
    [matches, scores] = vl_ubcmatch(des1, des2) ;

    matchedI1 = pts1(1:2,matches(1,:));
    matchedI2 = pts2(1:2,matches(2,:));
    
%     figure(2)
%     showMatchedFeatures(img1, img2, matchedI1', matchedI2')
%     saveas(gcf,'SIFTMatching.png')
    
%     subplot(1,2,1);
%     imshow(uint8(img1));
%     hold on;
%     plot(pts1(1,matches(1,:)),pts1(2,matches(1,:)),'b*');
% 
%     subplot(1,2,2);
%     imshow(uint8(img2));
%     hold on;
%     plot(pts2(1,matches(2,:)),pts2(2,matches(2,:)),'r*');
%     truesize;

  
%     matches = zeros(size(des1, 1), 2);
%     matches(:,1) = 1:size(des1, 1);
% 
%     [idx, d] = knnsearch(des2, des1, 'k', 2, 'NSMethod', 'kdtree');
%     matches(:,2) = idx(:,1);
%     
%     confidences = d(:,1) ./ d(:,2);
% 
%     [confidences, ind] = sort(confidences, 'ascend');
%     cidx = find((confidences<0.8)==0, 1);
%     matches = matches(ind, :);
%     matches = matches(1:cidx, :);
%     matchedI1 = pts1(matches(:,1));
%     matchedI2 = pts2(matches(:,2));
%     
%     finalKp1 = matchedI1.Location;
%     finalKp2 = matchedI2.Location;
%     
%     figure(2)
%     showMatchedFeatures(img1, img2, matchedI1, matchedI2)
%     saveas(gcf,'SIFTMatching.png')
end