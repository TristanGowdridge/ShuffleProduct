function f = partial_fractions(filename, precision)
syms x0 a1 a2 a b k1 k2 k3 A %  These will be used in the evals.

PARALLEL_CUT_IN = 20;

% If a file of this name already exists, delete it to not cause repeats in
% the data.
if exist(strcat(filename, '_MATLAB.txt'), 'file') == 2
    delete(strcat(filename, '_MATLAB.txt'));
end
fid = fopen(strcat(filename, '_python.txt'));

% Get the number of lines in the file for preallocating the cell.
n = 0;
while ~feof(fid)
    fgetl(fid);
    n = n+1;
end
fclose(fid);

if (n < PARALLEL_CUT_IN) % If less than don't use parallel.
    fid = fopen(strcat(filename, '_python.txt'), 'r');
    while ~feof(fid)
        term = eval(fgetl(fid));
        if isa(term, 'sym')
            term = partfrac(term, x0);
        end
        if precision
            term = vpa(term, precision);
        end
        term_str = string(term).replace('^', "**").replace('i', 'j');
        writelines(term_str, strcat(filename, "_MATLAB.txt"), WriteMode="append");
    end
    fclose(fid);


else % Parallel.
    % Evaluate and assign each line to the cell.
    fid = fopen(strcat(filename, '_python.txt'), 'r');
    line_list = cell(n, 1);
    index = 1;
    while ~feof(fid)
        line_list{index} = eval(fgetl(fid));
        index = index + 1;
    end
    fclose(fid);

    % Loop over the length of line_list and assign the partial fractions
    % expansion to each index.
    parpool('Processes')
    parfor i =1:n
        term = line_list{i};
        if isa(term, 'sym')
            term = partfrac(term, x0);
        end
        if precision
            term = vpa(term, precision);
        end
        line_list{i} = term;
    end
    delete(gcp('nocreate'));

    % Write to the file, so that these can be loaded into Python.
    for term = line_list
        term_str = string(term).replace('^', "**").replace('i', 'j');
        writelines(term_str, strcat(filename, "_MATLAB.txt"), WriteMode="append");
    end
end
end

