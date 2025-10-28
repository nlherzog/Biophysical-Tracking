function fit_lin_gyration_V3_modi_Tong(dt,conv,D0,min_track_length_lin,filename)%,filename

 %[filename,path] = uigetfile('multiselect','on','.mat');
 %cd(path)

min_track_length_expo = 50;
sliding_size = 30; % sliding window size for selected trajectories not MSD
sliding_step = 1;

flin = fittype('a*x');
fexp = fittype('a^2*(1-exp(-b*x))');

h = waitbar(0,'Fitting and extracting data...');
for k = 1:length(filename)

        result = struct();
        disp(filename{k})
           if exist(fullfile(cd, filename{k}), 'file') == 0
                continue
           elseif filename{k} == "Cell_placeholder"
                continue
           end


        result = importdata(filename{k});

        num_tracks = size(result,1);
        if num_tracks == 1
           num_tracks = size(result,2);
        end

        %% Linear fit of all the trajectories that have a min length > min_track_length_lin
         %exist(fullfile(cd, 'tracked_Traj_atg13_rep2_3_2.tif.mat'), 'file')
        isolate_idx = [];
        track_length = zeros(1,num_tracks);
        if min_track_length_lin < 11
            min_track_length_lin = 11;
        end

        for j = 1:num_tracks
            if length(result(j).tracking.x) >= min_track_length_lin
                isolate_idx = [isolate_idx;j];
            end
            track_length(j) = length(result(j).tracking.x);
        end

        size_rest_data = length(isolate_idx);


        lin_fit = cell(size_rest_data,7); % changed second value to 7 because I added 3 more values (x and y centroid and track length)

        Dlin = zeros(1,size_rest_data);
        Dlin_err = zeros(1,size_rest_data);
        Dlin_gof_rsquare = zeros(1,size_rest_data);
        Dlin_gof_rmse = zeros(1,size_rest_data);
        Dlin_centroid_x = cell(1,size_rest_data);
        Dlin_centroid_x(:,:) = {0};
        Dlin_centroid_y = cell(1,size_rest_data);
        Dlin_centroid_y(:,:) = {0};
        
       


        parfor j = 1:size_rest_data
        %for j = 1:size_rest_data

            time = result(isolate_idx(j)).tracking.time;
            MSD = result(isolate_idx(j)).tracking.MSD;


            Dlin_centroid_x{1,j} = result(isolate_idx(j)).tracking.x;
            Dlin_centroid_y{1,j} = result(isolate_idx(j)).tracking.y;

            [yy_lin,gof] = fit(time(1:10),MSD(1:10),flin,...
                        'display','off','Startpoint',D0);

            Dlin(j) = yy_lin.a/4;
            err = confint(yy_lin);
            Dlin_err(j) = err(2)-err(1);
            Dlin_gof_rsquare(j) = gof.rsquare;
            Dlin_gof_rmse(j) = gof.rmse;



        end

        lin_fit = {{Dlin},{Dlin_err},{Dlin_gof_rsquare},{Dlin_gof_rmse},{Dlin_centroid_x},{Dlin_centroid_y}, {track_length}};


        %% Corraled analysis of all the trajectories that have a min length > min_track_length_expo

        isolate_idx = [];

        if min_track_length_expo < min_track_length_lin
            min_track_length_expo = min_trakc_length_lin;
        end

        for j = 1:num_tracks
            if length(result(j).tracking.x) >= min_track_length_expo
                isolate_idx = [isolate_idx;j];
            end
        end

        size_rest_data = length(isolate_idx);

        corr_fit = cell(size_rest_data,3);



        parfor j = 1:size_rest_data
%         for j = 1:size_rest_data        

            Traj_length = length(result(isolate_idx(j)).tracking.x);
            chunks_number = floor((Traj_length-sliding_size)/sliding_step);
            Dexpo = zeros(1,chunks_number);
            R_c = zeros(1,chunks_number);


            for i = 1:chunks_number
                t_res(i) = result(isolate_idx(j)).tracking.time(1+(i-1)*sliding_step);
            end


            MSD_sliding = zeros(sliding_size-1,chunks_number);
            t_res = zeros(1,chunks_number);

            for i = 1:chunks_number
        %         if exist_z == 0
                    MSD_sliding(:,i) = calculate_MSD_modi_Tong(result(isolate_idx(j)).tracking.x(1+(i-1)*sliding_step:(i-1)*sliding_step + sliding_size),...
                        result(isolate_idx(j)).tracking.y(1+(i-1)*sliding_step:(i-1)*sliding_step + sliding_size),...
                        0,dt,conv);
                    MSD_sliding(:,i) = MSD_sliding(:,i) - MSD_sliding(1,i);
        %         elseif exist_z == 1
        %             MSD_sliding(:,i) = calculate_MSD(result(isolate_idx(j)).tracking.x(1+(i-1)*sliding_step:(i-1)*sliding_step + sliding_size),...
        %                 result(isolate_idx(j)).tracking.y(1+(i-1)*sliding_step:(i-1)*sliding_step + sliding_size),...
        %                 result(isolate_idx(j)).tracking.z(1+(i-1)*sliding_step:(i-1)*sliding_step + sliding_size),...
        %                 dt,conv);
        %             MSD_sliding(:,i) = MSD_sliding(:,i) - MSD_sliding(1,i);
        %         end
            end




            for i = 1:chunks_number

                t = result(isolate_idx(j)).tracking.time(1+(i-1)*sliding_step:(i-1)*sliding_step+sliding_size);
                t_fit = t-t(1);

                x = conv*result(isolate_idx(j)).tracking.x(1+(i-1)*sliding_step:(i-1)*sliding_step+sliding_size);
                y = conv*result(isolate_idx(j)).tracking.y(1+(i-1)*sliding_step:(i-1)*sliding_step+sliding_size);

                mean_x = mean(x);
                mean_y = mean(y);

                Rg = sqrt(numel(x)^(-1)*(sum((x-mean_x).^2 + (y-mean_y).^2)));

                [yy_lin] = fit(t_fit(1:11),...
                    MSD_sliding(1:11,i),flin,'StartPoint',D0,'display','off');

                    R_c(i) = Rg;

                    Dexpo(i) = yy_lin.a/4;


            end

            corr_fit(j,:) = {{Dexpo},{R_c},{isolate_idx(j)}};

        end

        %% Calculating the ensemble average MSD and the mean-time average MSD

        isolate_idx = [];
        size_track = [];


        if min_track_length_lin < 11
            min_track_length_lin = 11;
        end

        for j = 1:num_tracks

           if length(result(j).tracking.x) >= min_track_length_lin
                isolate_idx = [isolate_idx;j];
                size_track = [size_track;length(result(j).tracking.x)]; 
           end
           
        end
        
        size_rest_data = length(isolate_idx);
 
        max_track = max(size_track);
        MSD_e = zeros(1,max_track);
        MSD_t = zeros(1,max_track);
        x_MSD_ens = zeros(size_rest_data,max_track);
        y_MSD_ens = zeros(size_rest_data,max_track);
        MSD_MSD_t = zeros(size_rest_data,max_track);

        for i = 1:size_rest_data
            x_MSD_ens(i,1:size_track(i)) = result(isolate_idx(i)).tracking.x(1:end);
            y_MSD_ens(i,1:size_track(i)) = result(isolate_idx(i)).tracking.y(1:size_track(i));
            MSD_MSD_t(i,1:size_track(i)-1) = result(isolate_idx(i)).tracking.MSD(1:size_track(i)-1);
        end

        for i = 1:max_track
            idx = find(x_MSD_ens(:,i) > 0);
            MSD_e(i) = conv^2*mean( ( (x_MSD_ens(idx,i) - x_MSD_ens(idx,1)).^2 + (y_MSD_ens(idx,i) - y_MSD_ens(idx,1)).^2 ) );

            idx = find(MSD_MSD_t(:,i) > 0);
            MSD_t(i) = mean(MSD_MSD_t(idx,i));
        end

        MSD = {{MSD_e},{MSD_t}}; %MSD_e is the ensemble-averaged MSD, MSD_t is ensemble_average and time_averaged MSD

        % %% Calculating the ergodicity
        % 
        % isolate_idx = [];
        % 
        % for j = 1:num_tracks
        %    if length(result(j).tracking.x) >= Tergo
        %         isolate_idx = [isolate_idx;j];
        %    end
        % end
        % 
        % size_rest_data = length(isolate_idx);
        % 
        % eps = zeros(size_rest_data,Tergo);
        % 
        % for i = 1:size_rest_data
        %     eps(i,1:Tergo) = result(isolate_idx(i)).tracking.MSD(1:Tergo)./MSD_e(1:Tergo)';
        % end
        % 
        % ergo = {eps};
        %% Greg - attempting to calculate ensemble average MSD with weights.
        
        %%  Variance versus tau?     
        j = [];
        isolate_idx = [];
        track_length = zeros(1,num_tracks);
        

        
        if min_track_length_lin < 11
            min_track_length_lin = 11;
        end

        for j = 1:num_tracks
            if length(result(j).tracking.x) >= min_track_length_lin
                isolate_idx = [isolate_idx;j];
            end
            track_length(j) = length(result(j).tracking.x);
        end
        j = [];
        size_rest_data = length(isolate_idx);
        
        

        MSD_std = zeros(1,size_rest_data);
        time = zeros(1,size_rest_data);
        
        hold all
        
        
        for j = 1:size_rest_data
         MSD_std = result(isolate_idx(j)).tracking.std_MSD;
         MSD_var(1:length(MSD_std),j) = MSD_std.^2;
         time(1:length(MSD_std),j) = result(isolate_idx(j)).tracking.time;
        end
        
        t_step = time(3,1) - time(2,1);
        
        
        
        MSD_var(isnan(MSD_var)) = 0;
        time_array = 0:t_step:(t_step * length(MSD_var(:,1)));
        
        l = [];
        mean_MSD_var = zeros(1,size_rest_data);
        length(MSD_var(:,1))
        for l = 1:length(MSD_var(:,1))
            mean_MSD_var(l) = mean(MSD_var(l,MSD_var(l,:) > 0));
        end
        %t = time(1:length(mean_MSD_var),1);
        %top = 1:sum(t > 0);
        hold all
        plot(time_array(1,1:length(MSD_var(:,1))),mean_MSD_var(1,1:length(MSD_var(:,1))))
        % now remove NaN and find mean variance across rows
        % take largest time step and use that for the time to plot? 
        %Time2 = 
        %MSD_var_mean = 
        %clear t
        clear mean_MSD_var
        clear time
        clear top

        %% Calculating the displacement distribution and orientation distribution

        isolate_idx = [];

        for j = 1:num_tracks
           if length(result(j).tracking.x) >=  min_track_length_lin
                isolate_idx = [isolate_idx;j];
           end
        end

        Tergo = 11;  %This is the maximum time differences in steps
        size_rest_data = length(isolate_idx);
        disp_res = cell(1,Tergo);
        disp_ori = cell(1,Tergo);

        displ = [];
        ori = [];

        for i = 1:Tergo
            for j = 1:size_rest_data  
                displ = [displ;conv*displacement(result(isolate_idx(j)).tracking.x,result(isolate_idx(j)).tracking.y,i)'];
                ori = [ori;disp_corr(result(isolate_idx(j)).tracking.x,result(isolate_idx(j)).tracking.y,i)'];
            end
            disp_res{i} = {displ};
            disp_ori{i} = {ori};
        end



        %% Calculating the next step correlation

        next_step_res = [];
        step_next_size = [];

        for j = 1:size_rest_data 
            [next_step_calc,step_next_size_calc] = next_step(result(isolate_idx(j)).tracking.x,result(isolate_idx(j)).tracking.y);
            next_step_res = [next_step_res;conv*next_step_calc'];
            step_next_size = [step_next_size;conv*step_next_size_calc'];
        end

        next = {{next_step_res},{step_next_size}};


        %% Last step, saving the data
        lin_fit = {{Dlin},{Dlin_err},{Dlin_gof_rsquare},{Dlin_gof_rmse},{Dlin_centroid_x},{Dlin_centroid_y},{track_length}};

        row_headings = {'D_corr','R_c','track_index'};
        extract_corr = cell2struct(corr_fit,row_headings,2);
        row_headings = {'D_lin','D_lin_err','gof_rsquare','gof_rmse','Dlin_centroid_x','Dlin_centroid_y', 'track_length'};
        extract_lin = cell2struct(lin_fit,row_headings,2);
        row_headings = {'MSD_ens','MSD_time'};
        extract_MSD = cell2struct(MSD,row_headings,2);
        % extract_ergo = cell2struct(ergo,'ergo',1);
        extract_step = cell2struct(disp_res,'step',1);
        extract_ori = cell2struct(disp_ori,'ori',1);
        row_headings = {'corr','step_size'};
        extract_next = cell2struct(next,row_headings,2);

        extract = struct('lin',extract_lin,...
            'corr',extract_corr,...
            'MSD',extract_MSD,...
            'step',extract_step,...
            'ori',extract_ori,...
            'next',extract_next);

        saving_name = strcat('analyzed_',filename{k});

        save(saving_name,'extract')



        waitbar(k/length(filename))

 end


        disp('Extracted data saved')
        close(h)
  
end
