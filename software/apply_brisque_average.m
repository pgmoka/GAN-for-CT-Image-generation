% Precondition: string with path to directory to be analyzed
% Postcondition: int with the average of the BRISQUE
% between all png in the file path
function returnMe = apply_brisque_average(path)
    files_names = dir([path '*.png']);
    returnMe = 0;
    for k=1:length(files_names)
        name = [path files_names(k).name];
        ar = imread(name);
        % Adds to variable
        returnMe = returnMe + brisque(ar);
    end
    % Calculates average
    returnMe = returnMe / length(files_names);
end
