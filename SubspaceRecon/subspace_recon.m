%% load data
%setting--------------------------------------------------------------------------
clear; clc;close all;
%%
addpath(genpath('utils'))
disp('Loading data');
addpath(genpath('/cluster/berkin/berkin/Matlab_Code_New/LIBRARY/'));
addpath(genpath('/autofs/cluster/berkin/berkin/Matlab_Code_New/pulseq/pulseq-develop_1.4.0'))
addpath(genpath('/autofs/cluster/berkin/berkin/Matlab_Code/TOOLBOXES/SENSE_LSQR_Toolbox'))
addpath(genpath('/autofs/cluster/berkin/berkin/Matlab_Code_New/TOOLBOXES/CS_Wave_Toolbox'))
addpath '/autofs/cluster/berkin/berkin/Matlab_Code_New/NEATR/NEATR_Wip_936_Toolbox/neatr_wip_936_sense'
% addpath '/autofs/homes/003/sl1618/QALAS_optimization/ismrm2023_qalas_optimized-main/reconstruction_simulated_data/simulated_reconstruction/mwf_qalas/simulation/simulation_functions'
addpath '/autofs/homes/003/sl1618/QALAS_optimization/ismrm2023_qalas_optimized-main/reconstruction_simulated_data/simulated_reconstruction/utils'
% addpath('/homes/3/sl1618/works/bart-0.9.00')
% addpath('/homes/3/sl1618/works/bart-0.9.00/matlab')
addpath('inv/')
% % load sequence_flip_angles.mat
addpath(genpath('/autofs/cluster/berkin/yohan/python_code/bart-0.9.00'));
% setenv('BART_TOOLBOX_PATH', '/autofs/homes/003/sl1618/works/bart-0.9.00');
setenv('LD_LIBRARY_PATH', '/autofs/cluster/berkin/yohan/python_code/bart-0.9.00/lib:$LD_LIBRARY_PATH');
setenv('CUDA_BASE', '/usr/local/cuda/')
%setting--------------------------------------------------------------------------
%%
data_file_path = '/autofs/space/marduk_001/users/shizhuo/20250626_nist_bay4/20250626_nist/meas_MID00252_FID142118_pulseq_mgh_mwf_opt5_R5.dat';
load('OmniSeq.m','paras_epoch_1000')
num_acqs = 5; 
slices = 1:256;
K=4;
TE = paras_epoch_1000(end-1:end);
gap_between_readouts=paras_epoch_1000(641:645); %mwf op
alpha_train = paras_epoch_1000(1:640)*180/pi;

turbo_factor=128;E=num_acqs*turbo_factor;
[p,n,e] = fileparts(data_file_path);
basic_file_path = fullfile(p,n);

twix_obj = mapVBVD2(data_file_path);
data_unsorted_ = twix_obj{end}.image.unsorted();
[adc_len,ncoil,readouts] = size(data_unsorted_);

% Read params from seq file
pulseq_file_path = 'OmniSeq/opt5_R5.seq';

seq = mr.Sequence();
seq.read(pulseq_file_path);

N = seq.getDefinition('Matrix');
nTR = seq.getDefinition('nTR');
nETL = seq.getDefinition('nETL');

os_factor = seq.getDefinition('os_factor');

traj_y = seq.getDefinition('traj_y');
traj_z = seq.getDefinition('traj_z');
step_size = 5 * nETL;
%--------------------------------------------------------------------------
%% patref scan
%--------------------------------------------------------------------------
disp('Ref scan');

Ny_acs=32;
Nz_acs=32;


temp = 1:N(2);
iY_acs_indices = temp(1+end/2-Ny_acs/2:end/2+Ny_acs/2);

temp = 1:N(3);
iZ_acs_indices = temp(1+end/2-Nz_acs/2:end/2+Nz_acs/2);

ref = zeros(N(1),N(2),N(3),ncoil);

data_unsorted_ref = data_unsorted_(:,:,1:Ny_acs*Nz_acs);

index=1;
for iZ = iZ_acs_indices
    for iY = iY_acs_indices

            ref(:,iY,iZ,:) = data_unsorted_ref(:,:,index);    
            index=index+1;
    end
end

img_ref = ifft3call(ref);

imagesc3d2(rsos(img_ref,4), s(img_ref)/2, 1, [0,180,180], [-0,1e-3]), setGcf(.5)
%% do Reconstuction
%% loading sequences and setting sequence simulation parameters
M=240;N=208;C=32;
%-sequence parameters
TR         = 4500e-3;   % excluding dead time at the end of the sequence 
num_reps   = 5;       % number of repetitions to simulate to reach steady state
echo2use   = 1;
esp        = 5.74 * 1e-3;     % echo spacing in sec
turbo_fact = 128;  % ETL
time2relax_at_the_end = 0;  % actual TR on the console = TR + time2relax_at_the_end

%% setting up dictionary entries 
t1_entries  = [50:5:3000,3100:100:5000];
t2_entries  = [6:1:100,102:2:200,204:10:400,420:20:500];
b1_val      = 0.65:0.05:1.35;
b1_val_sub  = 0.65:0.05:1.35;
inv_eff     = 0.6:0.05:1.0;
inv_eff_sub = 0.6:0.05:1.0;

T1_entries = repmat(t1_entries.', [1,length(t2_entries)]).';
T1_entries = T1_entries(:);
  

T2_entries = repmat(t2_entries.', [1,length(t1_entries)]);
T2_entries = T2_entries(:);


t1t2_lut = cat(2, T1_entries, T2_entries);

% remove cases where T2>T1
idx = 0;
for t = 1:size(t1t2_lut,1)
    if t1t2_lut(t,1) < t1t2_lut(t,2)
        idx = idx+1;
    end
end

t1t2_lut_prune = zeross( [size(t1t2_lut,1) - idx, 2] );

idx = 0;
for t = 1:size(t1t2_lut,1)
    if t1t2_lut(t,1) >= t1t2_lut(t,2)
        idx = idx+1;
        t1t2_lut_prune(idx,:) = t1t2_lut(t,:);
    end
end
disp(['dictionary entries: ', num2str(size(t1t2_lut_prune,1))])

%% -generating dictionaries
length_b1_val       = length(b1_val);
length_b1_val_sub   = length(b1_val_sub);
length_inv_eff      = length(inv_eff);
length_inv_eff_sub  = length(inv_eff_sub);
cnt         = 0;
iniTime1    = clock;

signal_conv_fit     = zeross([length(t1t2_lut_prune), num_acqs, length(b1_val), length_inv_eff]);
signal_sub          = zeross([E, length(t1t2_lut_prune), length(b1_val_sub), length_inv_eff]);
signal_sub_fit      = zeross([E, length(t1t2_lut_prune), length(b1_val), length_inv_eff]);

% parallel computing
delete(gcp('nocreate'))
c = parcluster('local');
total_cores = c.NumWorkers;
parpool(min(ceil(total_cores*.5), length(b1_val)))

% for subspace
cnt         = 0;
iniTime     = clock;
iniTime0    = clock;
parfor b1 = 1:length_b1_val_sub
    for ie = 1:length_inv_eff_sub
        cnt             = cnt + 1;
       
        [Mz_all,Mxy_all] = sim_qalas_acqs_mwf(alpha_train, esp, turbo_fact, t1t2_lut_prune(:,1)*1e-3,...
                           t1t2_lut_prune(:,2)*1e-3, num_reps, num_acqs, gap_between_readouts, ...
                           b1_val(b1),inv_eff(ie),TE);       
        temp_sub        = squeeze(Mxy_all(:,:,end));    
        signal_sub(:,:,b1,ie)       = temp_sub;
        temp_sub_fit    = abs(Mxy_all(:,:,end));  
        for n = 1:size(temp_sub_fit,2)            
            temp_sub_fit(:,n)   = temp_sub_fit(:,n) / sum(abs(temp_sub_fit(:,n)).^2)^0.5;
        end
        signal_sub_fit(:,:,b1,ie)   = temp_sub_fit;
    end
end
delete(gcp('nocreate'))
fprintf('total elapsed time: %.1f sec\n\n',etime(clock,iniTime0));

% for subspace
signal_sub = reshape(signal_sub,[size(signal_sub,1), ...
                        size(signal_sub,2)*size(signal_sub,3)*size(signal_sub,4)])';
figure; plot(signal_sub(1:1e3:end,:)'); title('QALAS signals for subspace');

% for subspace fitting
length_dict     = length(t1t2_lut_prune);
dict_sub_fit    = zeross([length_dict * length_inv_eff, E, length(b1_val)]);
for t = 1:length_inv_eff
    dict_sub_fit(1 + (t-1)*length_dict : t*length_dict, :, :) = permute(signal_sub_fit(:,:,:,t),[2,1,3]);
end
figure; plot(dict_sub_fit(1:1e3:end,:)'); title('QALAS dictionary for subspace fitting');
fprintf('done\n')

%% -generating subspace
[u,s,v] = svd(signal_sub','econ'); 
phi     = reshape(u(:,1:K,:),1,1,1,1,1,E,K);
writecfl('data_nist/phi',phi)

%% Subspace reconstruction
num_slice = length(slices);
t2estimates_sub = zeros([M,N,num_slice]);
t1estimates_sub = zeros([M,N,num_slice]);
pdestimates_sub = zeros([M,N,num_slice]);
ieestimates_sub = zeros([M,N,num_slice]);
recs_save = zeros([M,N,K,num_slice]);
CC = 8;
for i =1:num_slice
    disp(['Running slice: ' num2str(i)]);
    slice = slices(i);
    patref_pad = mifft(padarray( ref, [0, 0, 0, 0]/2 ),1);
    patref_slice        = reshape(squeeze(patref_pad(slice,:,:,:)),M,N,1,32);
    fprintf('\nSVD coil compresion... ');
    comp_mtx    = squeeze(bart(sprintf('cc -M -p %d -S',CC),patref_slice)); % org code
    
    data_unsorted = permute(data_unsorted_,[1,3,2]);
    data_unsorted = reshape((comp_mtx(:,1:CC)'...
            *reshape(data_unsorted,adc_len*readouts,32).').',adc_len,readouts,CC);
    data_unsorted = permute(data_unsorted,[1,3,2]);
    patref_cc   = reshape((comp_mtx(:,1:CC)'...
            *reshape(patref_slice,M*N,32).').',M,N,1,CC);
    coils           = squeeze(bart('ecalib -m 1',patref_cc));
    coils_yh = coils;
    coils_bart = reshape(coils_yh,M,N,1,CC);
    
    % kspace
    data_unsorted_img = data_unsorted(:,:,Ny_acs*Nz_acs+1:end);    
    data_unsorted_img_1 = fftshift(ifft(ifftshift(data_unsorted_img,1),[],1),1)*sqrt(N(1));
    data_unsorted_slice = sq(data_unsorted_img_1(slice,:,:));
    kspace_2d = zeros([M N num_acqs*nETL CC]);
    for TR=1:nTR
        echo_num=1;
        for contrast=1:num_acqs
            index_start = (contrast-1)*nETL + (TR-1)*step_size +1;
            index_start_ro = (contrast-1)*nETL + (TR-1)*(num_acqs * nETL) +1;
            index_ro = index_start_ro;
            for index=(index_start):(index_start+nETL-1)
                ky = traj_y(index);
                kz = traj_z(index);
                kspace_2d(ky,kz,echo_num,:) = data_unsorted_slice(:,index_ro);
                index_ro = index_ro + 1;
                echo_num = echo_num + 1;
            end
        end
    end
    
    kspace_2d = reshape(kspace_2d,[1,M,N,num_acqs*nETL,CC]);
    kspace_2d = permute(kspace_2d, [1 2 3 5 4]);
    kspace = reshape(permute(sq(kspace_2d),[1 2 3 4]),[M,N,CC,E]);
    kspace  = kspace / norm(kspace(:));
    
    types        = 'wavelets';
    iters        = 1000;  %2000,800
    lambdas      = 2e-6; %2e-6,5e-6
    result = zeros([M,N,E,5]);
    if(strcmp(types, 'llr'))
        bartstr = sprintf('-R L:3:0:%f -i %d',lambdas,iters);
    else
        bartstr = sprintf('-l1 -r %f -i %d',lambdas,iters);
    end
    
    ksp = reshape(kspace,M,N,1,CC,1,E);
    coeff_sub = bart(sprintf('pics -g -S -w 1 -B data_nist/phi %s',bartstr),ksp,coils_bart);
    recs_save(:,:,:,i) = reshape(squeeze(coeff_sub),M,N,K);
    
    
end
delete(gcp('nocreate'))

s = 155;
figure;imagesc(abs(squeeze(recs_save(:,:,1,s))));

save('Omni_3d.mat','recs_save','-v7.3');


