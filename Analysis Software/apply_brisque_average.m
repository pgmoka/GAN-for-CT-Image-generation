
%  Made by Pedro Goncalves Mokarzel
%  while attending UW Bothell Student ID# 1576696
%  Made in 12/09/2019
%  Based on instruction in CSS 490, 
%  taught by professor Dong Si

% Precondition: string with path to directory to be analyzed
% Postcondition: int with the average of the BRISQUE,
% between all png in the file path, and BRISQUE score of each image as an
% array
function [returnMe, bris_list] = apply_brisque_average(path)
    files_names = dir([path '*.flt']);
    returnMe = 0;
    bris_list = [];
    T = struct2table(files_names);
    sortT = sortrows(T,'date');
    files_names = table2struct(sortT);
    for k=1:length(files_names)
        name = [path files_names(k).name];
        fid = fopen(name,'r');
        ar = fread(fid, 'float');
        ar = reshape(ar, 512,512);   %or other dimensions if you are reading a sinogram)
        fclose(fid);
        % Adds to variable
        bris =  brisque(ar);
        bris_list = [bris_list bris];
        returnMe = returnMe + bris;
    end
    % Calculates average
    returnMe = returnMe / length(files_names);
end
