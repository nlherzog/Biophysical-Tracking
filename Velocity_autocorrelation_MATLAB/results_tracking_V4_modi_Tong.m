function [filenames_all_conditions] = results_tracking_V4_modi_Tong(dt,conv, filenames_this_condition)

        for m = 1:length(filenames_this_condition)
            filename = filenames_this_condition{m}
            % import trajectory xlsx file exported from Trackmate
            if filename == "Cell_placeholder"
                continue
            end
            [Trajectory, Frame, x, y, z] = Read_Trackmate_traj_xlsx(filename) ;

            idx = find(diff(Trajectory) > 0);
            if isempty(idx)
                continue
            end
            
            S = [];
            for i = 1:length(idx)+1  % Total number of trajectories is "length(idx)+1"
                if i < 10
                    S = [S;strcat('part0000',num2str(i))];
                elseif i >= 10 && i < 100
                    S = [S;strcat('part000',num2str(i))];
                elseif i >= 100 && i < 999
                    S = [S;strcat('part00',num2str(i))];
                elseif i >=1000 && i < 9999
                    S = [S;strcat('part0',num2str(i))];
                elseif i >= 10000 && i < 99999
                    S = [S;strcat('part',num2str(i))];
                end
            end

            field = cellstr(S);

            result = cell2struct(field','tracking',1);

            MSD = zeros(1,length(idx)+1);
            time = zeros(1,length(idx)+1);
            x_res = zeros(1,length(idx)+1);
            y_res = zeros(1,length(idx)+1);
            frame_res = zeros(1,length(idx)+1);
            std_MSD = zeros(1,length(idx)+1);
            n_MSD = zeros(1,length(idx)+1);
            
           for i = 1:length(idx)+1
                
                % "Frame" index starts from 0 in raw csv files, while
                % "Trajectory","time" index etc. all start from 1.
                % Thus, add "+ones" in the saved "frame_res" variable to
                % avoid zero exclusion of meaningful "0th" frame.
                if i == 1
                    x_res(1:idx(i),i) = x(1:idx(i));
                    y_res(1:idx(i),i) = y(1:idx(i));
                    frame_res(1:idx(i),i) = Frame(1:idx(i))+1;
                    time(1:idx(i)-1,i) = dt:dt:dt*(idx(i)-1);
                    [MSD(1:(idx(i)-1),i), std_MSD(1:(idx(i)-1),i), n_MSD(1:(idx(i)-1),i)] = calculate_MSD_modi_Tong(x(1:idx(i)),y(1:idx(i)),0,dt,conv);
                    a = [1:idx(i),i];
                    disp(a)
                elseif i > 1 && i<length(idx)+1
                    x_res(1:(idx(i)-idx(i-1)),i) = x((idx(i-1)+1):idx(i));
                    y_res(1:(idx(i)-idx(i-1)),i) = y((idx(i-1)+1):idx(i));
                    frame_res(1:(idx(i)-idx(i-1)),i) = Frame((idx(i-1)+1):idx(i))+1;
                    time(1:(idx(i)-idx(i-1)-1),i) = dt:dt:dt*(idx(i)-idx(i-1)-1);
                    [MSD(1:(idx(i)-idx(i-1)-1),i), std_MSD(1:(idx(i)-idx(i-1)-1),i), n_MSD(1:(idx(i)-idx(i-1)-1),i)] = calculate_MSD_modi_Tong(x((idx(i-1)+1):idx(i)),y((idx(i-1)+1):idx(i)),0,dt,conv);
                elseif i == length(idx)+1
                    x_res(1:(length(Trajectory)-idx(i-1)),i) = x(idx(i-1)+1:length(Trajectory));
                    y_res(1:(length(Trajectory)-idx(i-1)),i) = y(idx(i-1)+1:length(Trajectory));
                    frame_res(1:(length(Trajectory)-idx(i-1)),i) = Frame(idx(i-1)+1:length(Trajectory))+1;
                    time(1:(length(Trajectory)-idx(i-1)-1),i) = dt:dt:dt*(length(Trajectory)-idx(i-1)-1);
                    [MSD(1:(length(Trajectory)-idx(i-1)-1),i), std_MSD(1:(length(Trajectory)-idx(i-1)-1),i), n_MSD(1:(length(Trajectory)-idx(i-1)-1),i)] = calculate_MSD_modi_Tong(x(idx(i-1)+1:length(Trajectory)),y(idx(i-1)+1:length(Trajectory)),0,dt,conv);                    
                end 
                
                if length(find(MSD(:,i))) + 1 == length(find(x_res(:,i)))
                    result(i).tracking = struct('time',time(find(MSD(:,i))>0,i),...
                        'x',x_res(find(x_res(:,i))>0,i),...
                        'y',y_res(find(y_res(:,i))>0,i),...
                        'MSD',MSD(find(MSD(:,i))>0,i),...
                        'frame',frame_res(find(frame_res(:,i))>0,i),...
                        'std_MSD',std_MSD(find(MSD(:,i))>0,i),...
                        'n_MSD',n_MSD(find(MSD(:,i))>0,i));
                else
                    x_res_length_temp = length(find(MSD(:,i)))+1;
                    result(i).tracking = struct('time',time(find(MSD(:,i))>0,i),...
                        'x',x_res(1:x_res_length_temp,i),...
                        'y',y_res(1:x_res_length_temp,i),...
                        'MSD',MSD(find(MSD(:,i))>0,i),...
                        'frame',frame_res(1:x_res_length_temp,i),...
                        'std_MSD',std_MSD(find(MSD(:,i))>0,i),...
                        'n_MSD',n_MSD(find(MSD(:,i))>0,i));
                end

            end

            filename_int = filenames_this_condition{m};
            name_file = filename_int(1:strfind(filename_int,'.xlsx')-1);

            saving_name = strcat('tracked_',name_file,'.mat');
            save(saving_name,'result')

        end
        %% 
    

    % uisave('result')
    % save P=8_bis.mat result 

    end




