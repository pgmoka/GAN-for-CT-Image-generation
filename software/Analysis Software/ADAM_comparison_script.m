%  Made by Pedro Goncalves Mokarzel
%  while attending UW Bothell Student ID# 1576696
%  Made in 12/09/2019
%  Based on instruction in CSS 490, 
%  taught by professor Dong Si

% Set up arrays
[av_ar_default av_default] = apply_brisque_average('./Case 1/default_ADAM/');
[av_ar_custom av_custom] = apply_brisque_average('./Case 1/custom_ADAM/');
real_im = apply_brisque_specific('./Case 1/00003641_img.flt');

[av_ar_default_0 av_default_0] = apply_brisque_average('./Case 0/default_ADAM/');
[av_ar_custom_0 av_custom_0] = apply_brisque_average('./Case 0/custom_ADAM/');
real_im_0 = apply_brisque_specific('./Case 0/00001392_img.flt');

[av_ar_default_2 av_default_2] = apply_brisque_average('./Case 2/default_ADAM/');
[av_ar_custom_2 av_custom_2] = apply_brisque_average('./Case 2/custom_ADAM/');
real_im_2 = apply_brisque_specific('./Case 2/00000853_img.flt');

[av_ar_default_3 av_default_3] = apply_brisque_average('./Case 3/Data/');
real_im_3 = apply_brisque_specific('./Case 3/00003680_img.flt');

[av_ar_default_4 av_default_4] = apply_brisque_average('./Case 4/Data/');
real_im_4 = apply_brisque_specific('./Case 4/00002662_img.flt');

[av_ar_default_5 av_default_5] = apply_brisque_average('./Case 5/Data/');
real_im_5 = apply_brisque_specific('./Case 5/00000338_img.flt');

% Set proper range for printings
x = [0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190];
x2 = [0 30 60 90 120 150 180];

% corrects BRISQUE
for i = 1:length(x)
    av_default(i) = av_default(i) - real_im;
    av_custom(i) = av_custom(i) - real_im;
    
    av_default_0(i) = av_default_0(i) - real_im_0;
    av_custom_0(i) = av_custom_0(i) - real_im_0;
    
    av_default_2(i) = av_default_2(i) - real_im_2;
    av_custom_2(i) = av_custom_2(i) - real_im_2;
end

% Prints images
figure(1);
plot(x,av_custom,x,av_default);
title('Comparison Between Custom and Default ADAM for Case 1')
legend({'Custom ADAM', 'Default ADAM'},'Location','southeast')
xlabel('epoch')
ylabel('BRISQUE Score Adjusted with Original Case');

figure(2);
plot(x,av_custom_0);
title('Comparison Between Custom and Default ADAM for Case 0')
legend({'Custom ADAM', 'Default ADAM'},'Location','southeast')
xlabel('epoch')
ylabel('BRISQUE Score Adjusted with Original Case');

figure(3);
plot(x,av_custom_2,x,av_default_2);
title('Comparison Between Custom and Default ADAM for Case 2')
legend({'Custom ADAM', 'Default ADAM'},'Location','southeast')
xlabel('epoch')
ylabel('BRISQUE Score Adjusted with Original Case');

% Comparison for 3000 CT Images data
for i = 1:length(x2)
    av_default_3(i) = av_default_3(i) - real_im_3;
    av_default_4(i) = av_default_4(i) - real_im_4;
    av_default_5(i) = av_default_5(i) - real_im_5;
end

figure(4)
plot(x2,av_default_3,x2,av_default_4,x2,av_default_5);
title('3000 Image Analisis')
legend({'Case 3', 'Case 4', 'Case 5'},'Location','southeast')
xlabel('epoch')
ylabel('BRISQUE Score Adjusted with Original Case');