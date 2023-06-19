function f = pf_dict(filename, keys)

x0 = sym("x0");
a1 = sym("a1");
a2 = sym("a2");
k1 = sym("k1");
k2 = sym("k2");
k3 = sym("k3");
A  = sym("A");

PARALLEL_CUT_IN = 20;

% If a file of this name already exists, delete it to not cause repeats in
% the data.
for key = keys
    if exist(strcat(filename, num2str(key), '_MATLAB.txt'), 'file') == 2
        delete(strcat(filename, num2str(key), '_MATLAB.txt'));
    end
    fid = fopen(strcat(filename, num2str(key),'_python.txt'));

    % Get the number of lines in the file for preallocating the cell.
    n = 0;
    while ~feof(fid)
        fgetl(fid);
        n = n+1;
    end
    fclose(fid);

    if (n < PARALLEL_CUT_IN) % If less than don't use parallel.
        fid = fopen(strcat(filename, num2str(key), '_python.txt'), 'r');
        while ~feof(fid)
            term = eval(fgetl(fid));
            if isa(term, 'sym')
                term = partfrac(term, x0);
            end
            term_str = string(term).replace('^', "**").replace('i', 'j');
            writelines(term_str, strcat(filename, num2str(key), "_MATLAB.txt"), WriteMode="append");
        end
        fclose(fid);


    else % Parallel.
        % Evaluate and assign each line to the cell.
        fid = fopen(strcat(filename, num2str(key), '_python.txt'), 'r');
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
            line_list{i} = term;
        end
        delete(gcp('nocreate'));

        % Write to the file, so that these can be loaded into Python.
        for term = line_list
            term_str = string(term).replace('^', "**").replace('i', 'j');
            writelines(term_str, strcat(filename, num2str(key), "_MATLAB.txt"), WriteMode="append");
        end
    end
end
end
