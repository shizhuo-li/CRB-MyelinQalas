function [Mz_mtx_all, Mxy_mtx_all] = sim_qalas_acqs_mwf(alpha_deg, esp, turbo_factor,...
    t1_vals, t2_vals, num_reps, num_acqs, gap_between_readouts, b1, inv_eff, TE, pd_map)

num_voxels = length(t1_vals);

if nargin < 14
    M0 = ones(num_voxels,1);
else
    M0 = pd_map(:);
end

if nargin < 12
    inv_eff = 1;
end

if nargin < 11
    b1 = 1;
end

if nargin < 10
    time2relax_at_the_end = 0;
end

if nargin < 9
    gap_between_readouts = 900e-3;
end

time2relax_at_the_end = 53.5e-3;
inverse_eff = @(M0, IE)  -1.*M0 .* IE;

relax_T2 = @(Mz, TE_T2prep, T2)  Mz .* exp(-TE_T2prep ./ T2);

relax_T2_b1 = @(Mz, TE_T2prep, T2, b1, T1)  Mz .* ( sin(b1 * pi/2).^2 .* exp(-TE_T2prep ./ T2) + cos(b1 * pi/2).^2 .* exp(-TE_T2prep ./ T1) );

relax_T1 = @(M0, Mz, delta_t, T1)  M0 - (M0 - Mz) .* exp(-delta_t ./ T1);


etl = turbo_factor * esp;

tf = turbo_factor;       % Turbo factor, or the number of echoes.
alpha = b1 * alpha_deg * pi/180;


% timings are based on: https://jcmr-online.biomedcentral.com/track/pdf/10.1186/s12968-014-0102-0.pdf

% BB: hack
% delT_M1_M2 = 109.7e-3;                % duration of T2 prep

% delT_M1_M2 = TE;  % joint_4acq(_g1)
TE1 = TE(1);  % 80e-3; 29.7
TE2 = TE(2);    % 32.5e-3; 89.7

% timings below are hard coded:
% delT_M0_M1 = 900e-3 - etl - 110e-3;
% delT_M0_M1 = 900e-3 - etl - delT_M1_M2;
delT_M0_M1_1 = gap_between_readouts(1) - etl - TE1;
delT_M0_M1_2 = gap_between_readouts(2) - etl - TE2;

delT_M2_M3 = etl;                   % duration of readout#1


% delT_M2_M6 = 900e-3;                % between two readouts
delT_M2_M6 = gap_between_readouts(3);                % between two readouts
delT_M4_M5 = 12.8e-3;               % inversion pulse


% delT_M5_M6 = 100e-3 - 6.4e-3;       % gap between end of inversion and start of readout#2
delT_M5_M6 = 100e-3 - 6.45e-3;       % gap between end of inversion and start of readout#2

delT_M3_M4 = delT_M2_M6 - delT_M2_M3 - delT_M4_M5 - delT_M5_M6;     % between end of readout#1 and start of inversion


delT_M6_M7 = etl;                   % duration of readout#2
% delT_M7_M8 = 900e-3 - etl;          % from end of readout#2 to  begin of readout#3
delT_M7_M8 = gap_between_readouts(4) - etl;          % from end of readout#2 to  begin of readout#3
delT_M8_M9 = etl;                   % duration of readout#3

% delT_M9_M10 = 900e-3 - etl;         % from end of readout#3 to  begin of readout#4
if num_acqs > 4
    delT_M9_M10 = gap_between_readouts(5) - etl;         % from end of readout#3 to  begin of readout#4
    delT_M10_M11 = etl;                 % duration of readout#4
end

if num_acqs > 5
    % delT_M11_M12 = 900e-3 - etl;        % from end of readout#4 to  begin of readout#5
    delT_M11_M12 = gap_between_readouts(6) - etl;        % from end of readout#4 to  begin of readout#5
    delT_M12_M13 = etl;                 % duration of readout#5
end

% total_event_duration = delT_M0_M1 + delT_M1_M2 + delT_M2_M3 + delT_M3_M4 + delT_M4_M5 + delT_M5_M6 + delT_M6_M7 + delT_M7_M8 + delT_M8_M9 + delT_M9_M10 + delT_M10_M11 + delT_M11_M12 + delT_M12_M13;
% 
% disp(['total event duration: ', num2str(total_event_duration), ' sec'])
% 
% 
% delT_M13_2end = max(TR - total_event_duration, 0);
% 
% if time2relax_at_the_end > 0
%     delT_M13_2end = delT_M13_2end + time2relax_at_the_end;
% end
% 
% disp(['time to relax at the end of TR: ', num2str(delT_M13_2end), ' sec'])

Mz_mtx_all  = zeross([num_acqs*turbo_factor,num_voxels,num_reps]);
Mxy_mtx_all = zeross([num_acqs*turbo_factor,num_voxels,num_reps]);

Mstart = M0;

% tic
for reps = 1:num_reps
    % disp(['repetition: ', num2str(reps)])
% TE1    
    M1 = relax_T1(M0, Mstart, delT_M0_M1_1, t1_vals);
        
    M2 = relax_T2_b1(M1, TE1 - 9.7e-3, t2_vals, b1, t1_vals);
     
    % acq1
    Mz = M2;
    Mxy = zeros(tf, num_voxels);

    time = 9.7e-3;

    echo_ctr = 1;
    acq_ctr  = 1;
    
    if(acq_ctr <= num_acqs)
        for q = 1:tf
            Mz = relax_T1(M0, Mz, time, t1_vals);
            
            Mxy_mtx_all(echo_ctr,:,reps) = sin(alpha(echo_ctr)) * Mz;

            Mz = cos(alpha(echo_ctr)) * Mz;

            Mz_mtx_all(echo_ctr,:,reps) = Mz;
            echo_ctr = echo_ctr + 1;

            time = esp;
        end
        
        acq_ctr = acq_ctr + 1;
    end

%%TE2
    M1 = relax_T1(M0, Mz, delT_M0_M1_2, t1_vals);
        
    M2 = relax_T2_b1(M1, TE2 - 9.7e-3, t2_vals, b1, t1_vals);
     
    % acq1
    Mz = M2;

    time = 9.7e-3;
    
    if(acq_ctr <= num_acqs)
        for q = 1:tf
            Mz = relax_T1(M0, Mz, time, t1_vals);
            
            Mxy_mtx_all(echo_ctr,:,reps) = sin(alpha(echo_ctr)) * Mz;

            Mz = cos(alpha(echo_ctr)) * Mz;

            Mz_mtx_all(echo_ctr,:,reps) = Mz;
            echo_ctr = echo_ctr + 1;

            time = esp;
        end
        
        acq_ctr = acq_ctr + 1;
    end
    
    if(acq_ctr <= num_acqs)
        M3 = Mz;
        Mxy_acq1 = Mxy;

        M4 = relax_T1(M0, M3, delT_M3_M4, t1_vals);

        % inversion efficiency 
        M5 = -M4 .* inv_eff;
        % M5 = inverse_eff(M4,inv_eff);

        M6 = relax_T1(M0, M5, delT_M5_M6, t1_vals);

        % acq2
        Mz = M6;
        Mxy = zeros(tf, num_voxels);

        time = 0; 

        for q = 1:tf
            Mz = relax_T1(M0, Mz, time, t1_vals);

            Mxy_mtx_all(echo_ctr,:,reps) = sin(alpha(echo_ctr)) * Mz;

            Mz = cos(alpha(echo_ctr)) * Mz;

            Mz_mtx_all(echo_ctr,:,reps) = Mz;
            echo_ctr = echo_ctr + 1;

            time = esp;
        end
        acq_ctr = acq_ctr + 1;
    end
    
    if(acq_ctr <= num_acqs)
 
        M7 = Mz;
        Mxy_acq2 = Mxy; 

        M8 = relax_T1(M0, M7, delT_M7_M8, t1_vals);


        % acq3
        Mz = M8;
        Mxy = zeros(tf, num_voxels);

        time = 0;

        for q = 1:tf
            Mz = relax_T1(M0, Mz, time, t1_vals);

            Mxy_mtx_all(echo_ctr,:,reps) = sin(alpha(echo_ctr)) * Mz;

            Mz = cos(alpha(echo_ctr)) * Mz;

            Mz_mtx_all(echo_ctr,:,reps) = Mz;
            echo_ctr = echo_ctr + 1;

            time = esp;
        end
        acq_ctr = acq_ctr + 1;
    end
    
    if(acq_ctr <= num_acqs)
 
        M9 = Mz;
        Mxy_acq3 = Mxy;

        M10 = relax_T1(M0, M9, delT_M9_M10, t1_vals);

        % acq4
        Mz = M10;
        Mxy = zeros(tf, num_voxels);

        time = 0;

        for q = 1:tf
            Mz = relax_T1(M0, Mz, time, t1_vals);

            Mxy_mtx_all(echo_ctr,:,reps) = sin(alpha(echo_ctr)) * Mz;

            Mz = cos(alpha(echo_ctr)) * Mz;

            Mz_mtx_all(echo_ctr,:,reps) = Mz;
            echo_ctr = echo_ctr + 1;

            time = esp;
        end
        acq_ctr = acq_ctr + 1;
    end
    
    if(acq_ctr <= num_acqs)
        M11 = Mz;
        Mxy_acq4 = Mxy;


        M12 = relax_T1(M0, M11, delT_M11_M12, t1_vals);


        % acq5
        Mz = M12;
        Mxy = zeros(tf, num_voxels);

        time = 0;

        for q = 1:tf
            Mz = relax_T1(M0, Mz, time, t1_vals);

            Mxy_mtx_all(echo_ctr,:,reps) = sin(alpha(echo_ctr)) * Mz;

            Mz = cos(alpha(echo_ctr)) * Mz;
            Mz_mtx_all(echo_ctr,:,reps) = Mz;
            echo_ctr = echo_ctr + 1;

            time = esp;
        end
    end    
    
    Mz = relax_T1(M0, Mz, time2relax_at_the_end, t1_vals);

    Mstart = Mz;
    
    % disp(['mean Mz: ', num2str(mean(Mstart))])
    
end
% toc



end