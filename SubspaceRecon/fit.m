clear clc
%%
addpath inv
addpath(fullfile('mwf/utils'));
num_acqs = 5; E = num_acqs * 128;turbo_factor=128;
% load('/autofs/cluster/berkin/shizhuo/result_invivo/mwf_0331/3d/opt5_invr_3d.mat','recs_save')
load('/autofs/space/marduk_001/users/shizhuo/Results/invivo/0710/Omni_R5_3d.mat','recs_save')
% load('/autofs/space/marduk_001/users/shizhuo/Results/Phantom/0626/opt5R5_invr_3d.mat','recs_save')
addpath(genpath('mwf/'))
addpath('/autofs/homes/003/sl1618/QALAS_optimization/ismrm2023_qalas_optimized-main/reconstruction_simulated_data/simulated_reconstruction/mwf_qalas/simulation')
recs_save = permute(recs_save,[4,1,2,3]);
%% run- other view
slice = 100; % 165 130 90
coeff_sub = squeeze(recs_save(:,:,slice,:));
% coeff_sub   = recs_save;
% figure;imagesc(squeeze(abs(recs_save(:,:,110,1))));
M = size(coeff_sub,1);
N = size(coeff_sub,2);
recs_sub = reshape((squeeze(phi)*reshape(squeeze(coeff_sub),M*N,K).').',M,N,E);
% load('b1_0623.mat')
% b1_map = b1map_afi;
% b1_map = ones([M,N]);
% b1_map = b1map_afi_165x;
% b1_map = flipud(b1_map);
disp('loading done')
%% subspace dictionary setting
% load('sequence/mwf_joint_L20_i1000_wgw_5ro.mat','output_epoch_1000')
load('sequence/mwf_5ro_40_up.mat','output_epoch_1000')
% slices = 1:256;%
K=4;
% opt
gap_between_readouts=output_epoch_1000(641:645); %mwf op
alpha_train = output_epoch_1000(1:640)*180/pi;
TE = output_epoch_1000(end-1:end);
% conv
% TE = [29.7e-3, 89.7e-3];
% gap_between_readouts=[0.84442, 0.84442, 0.9, 0.9, 0.9, 0.9];
% alpha_train = ones(num_acqs*turbo_factor)*4;
% qalas
% TE = 109.7e-3;
% gap_between_readouts=[0.9, 0.9, 0.9, 0.9, 0.9];
% alpha_train = ones(num_acqs*turbo_factor)*4;

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
% b1_val      = 1;
% b1_val_sub  = 1;
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

%% generate subspace dictionary
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

% % for subspace
cnt         = 0;
iniTime     = clock;
iniTime0    = clock;
for b1 = 1:length_b1_val_sub
% for b1 = 1:length_b1_val_sub
    for ie = 1:length_inv_eff_sub
        cnt             = cnt + 1;
        % opt & conv
        % [Mz_all,Mxy_all] = sim_qalas_acqs_mwf_invrange(TR, alpha_train, esp, turbo_fact, t1t2_lut_prune(:,1)*1e-3,...
        %                    t1t2_lut_prune(:,2)*1e-3, num_reps, num_acqs, gap_between_readouts, time2relax_at_the_end,...
        %                    b1_val(b1),inv_eff(ie),TE);  
        % qalas

        [Mz_all,Mxy_all] = sim_qalas_allro_joint_inv(TR, alpha_train, esp, turbo_factor, t1t2_lut_prune(:,1)*1e-3,...
                            t1t2_lut_prune(:,2)*1e-3, num_reps, echo2use, gap_between_readouts, time2relax_at_the_end,...
                            b1_val(b1),inv_eff(ie),num_acqs, TE);

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
% figure; plot(signal_sub(1:1e3:end,:)'); title('QALAS signals for subspace');

% for subspace fitting
length_dict     = length(t1t2_lut_prune);
dict_sub_fit    = zeross([length_dict * length_inv_eff, E, length(b1_val)]);
for t = 1:length_inv_eff
    dict_sub_fit(1 + (t-1)*length_dict : t*length_dict, :, :) = permute(signal_sub_fit(:,:,:,t),[2,1,3]);
end
% figure; plot(dict_sub_fit(1:1e3:end,:)'); title('QALAS dictionary for subspace fitting');

fprintf('done\n')

%% -generating subspace
[u,s,v] = svd(signal_sub','econ'); 
phi     = reshape(u(:,1:K,:),1,1,1,1,1,E,K);
% writecfl('data_opt/phi',phi)
save('data/data_qalas_bi.mat','phi')
recs_sub = reshape((squeeze(phi)*reshape(squeeze(coeff_sub),M*N,K).').',M,N,E);
%% rub -other view
t2estimates_sub = zeros([M,N,1]);
t1estimates_sub = zeros([M,N,1]);
pdestimates_sub = zeros([M,N,1]);
ieestimates_sub = zeros([M,N,1]);

estimate_pd_map = 1;
% b1_map          = ones([M,N]);

% load('b1/b1_0710nist.mat');b1_map = b1map_afi;
load('b1/b1_0710.mat');b1_map = squeeze(b1(slice,:,:));
% b1_map = imresize(b1_map, [288, 256], 'bicubic');
% b1_map = ones([M,N]);
[T1_map_sub,T2_map_sub,PD_map_sub,IE_map_sub] = fit_dict_sub_mwf_3d(permute(recs_sub,[1,2,4,3]), ...
                                                    dict_sub_fit, t1t2_lut_prune, estimate_pd_map, ...
                                                    b1_map, inv_eff, ...
                                                    TR, alpha_train, esp, turbo_factor, ...
                                                    num_reps, echo2use, gap_between_readouts, ...
                                                    time2relax_at_the_end, num_acqs,TE);
fprintf('done\n')
   
t2estimates_sub(:,:,1) = T2_map_sub;
t1estimates_sub(:,:,1) = T1_map_sub;
pdestimates_sub(:,:,1) = PD_map_sub;
ieestimates_sub(:,:,1) = IE_map_sub;
fprintf('done\n')


s = 1;
mask = squeeze(abs(coeff_sub(:,:,1))>0.007);
% mask(248:256,:)=0;mask(240:256,1:61)=0;mask(240:256,149:208)=0; % y
% mask(248:256,:)=0;mask(240:256,1:85)=0;mask(240:256,181:240)=0; % z
figure;imagesc(squeeze(t1estimates_sub(:,:,s)).*mask,[0,3000]);colormap('hot');colorbar;%,[0,5500]
figure;imagesc(t2estimates_sub(:,:,s).*mask,[0,200]);colormap('hot');colorbar;%,[0,600]
% save('/autofs/space/marduk_001/users/shizhuo/Results/invivo/0710/Omni_R7790_lam4.mat','t1estimates_sub','t2estimates_sub')

%%===============================================
%% water fraction
%%===============================================
t1_entries  = [150,1100,1300,4500]; % 
t2_entries  = [20,70,80,500]; % 
b1_val      = 0.65:0.05:1.35; % 1 (120423 ver.)
b1_val_sub  = 0.65:0.05:1.35; % 1 (120423 ver.)
length_inv_eff      = 1;
length_inv_eff_sub  = 1;

T1_entries  = t1_entries; % PV
T1_entries  = T1_entries(:);

T2_entries  = t2_entries; % PV
T2_entries  = T2_entries(:);

t1t2_lut    = cat(2, T1_entries, T2_entries);

% remove cases where T2>T1
idx = 0;
for t = 1:length(t1t2_lut)
    if t1t2_lut(t,1) < t1t2_lut(t,2)
        idx = idx+1;
    end
end

t1t2_lut_prune = zeross([length(t1t2_lut) - idx, 2]);

idx = 0;
for t = 1:length(t1t2_lut)
    if t1t2_lut(t,1) >= t1t2_lut(t,2)
        idx = idx+1;
        t1t2_lut_prune(idx,:) = t1t2_lut(t,:);
    end 
end
% inv_eff = [0.9822;0.9827;0.9828;0.9829];
inv_eff = [0.5912;0.8546;0.8698;0.9636];

disp('inv generating --> done')

fprintf('dictionary size: %d \n', length(t1t2_lut_prune)*length(b1_val)*length_inv_eff);

E = turbo_factor * num_acqs;

signal_sub          = zeross([E, length(t1t2_lut_prune), length(b1_val_sub), length_inv_eff ]);
signal_sub_fit      = zeross([E, length(t1t2_lut_prune), length(b1_val), length_inv_eff]);

length_b1_val       = length(b1_val);
length_b1_val_sub   = length(b1_val_sub);

% for fitting
cnt         = 0;
iniTime1    = clock;
for b1 = 1:length_b1_val
    for ie = 1:length_inv_eff
        cnt             = cnt + 1;
        %%%conv and opt
        [Mz_,Mxy_] = sim_qalas_acqs_mwf_inv(TR, alpha_train, esp, turbo_factor, t1t2_lut_prune(:,1)*1e-3,...
                           t1t2_lut_prune(:,2)*1e-3, num_reps, num_acqs, gap_between_readouts, time2relax_at_the_end,...
                           b1_val(b1),inv_eff,TE);

        %%%%qalas

        % [Mz_,Mxy_] = sim_qalas_allro_joint_inv(TR, alpha_train, esp, turbo_factor, t1t2_lut_prune(:,1)*1e-3,...
        %                     t1t2_lut_prune(:,2)*1e-3, num_reps, echo2use, gap_between_readouts, time2relax_at_the_end,...
        %                     b1_val(b1),inv_eff,num_acqs, TE);
      
        temp_sub       = abs(Mxy_(:,:,end))';
        
        for n = 1:size(temp_sub,1)
            temp_sub(n,:)      = temp_sub(n,:) / sum(abs(temp_sub(n,:)).^2)^0.5;
        end
        
        signal_sub_fit(:,:,b1,ie)  = temp_sub';

    end
end
delete(gcp('nocreate'))
fprintf('total elapsed time: %.1f sec\n\n',etime(clock,iniTime1));

% for conventional fitting
length_dict = length(t1t2_lut_prune);
dict_sub    = zeross([length_dict * length_inv_eff , num_acqs*turbo_factor, length(b1_val)]);
for t = 1:length_inv_eff 
    dict_sub(1 + (t-1)*length_dict : t*length_dict, :, :) = permute(signal_sub_fit(:,:,:,t),[2,1,3]);
end

fprintf('done\n')


% PV dictionary-based
% dict_sub = reshape(dict_sub,[length_dict,num_acqs*turbo_factor*length(b1_val)]);
dict_sub_    = permute(dict_sub,[2,1,3]);
% dict_pv      = inv(dict_sub_.' * dict_sub_) * dict_sub_.';



PV_dict_1   = 0.02:0.02:0.5; % myelin water
% PV_dict_1   = 0.05:0.05:1.0; % myelin water
PV_dict_2   = 0.05:0.05:1.0; % intra cellular water
PV_dict_3   = 0.05:0.05:1.0; % extra cellular water
PV_dict_4   = 0.00:0.05:1.0; % free water

PV_dict     = zeros(4,length(PV_dict_1)*length(PV_dict_2)*length(PV_dict_3));

cnt = 0;

for pp = 1:length(PV_dict_1)
    for qq = 1:length(PV_dict_2)
        for rr = 1:length(PV_dict_3)
            for ss = 1:length(PV_dict_4)
                cnt = cnt + 1;
                    PV_dict(:,cnt) = [PV_dict_1(pp),PV_dict_2(qq),PV_dict_3(rr),PV_dict_4(ss)]./ ...
                        (PV_dict_1(pp)+PV_dict_2(qq)+PV_dict_3(rr)+PV_dict_4(ss));
            end
        end
    end
end
PV_dict = unique(PV_dict','rows')';
dict_pv_full=zeros(num_acqs*turbo_factor, size(PV_dict,2), length(b1_val));
for bb=1:length(b1_val)
    dict_pv_bb = squeeze(dict_sub_(:,:,bb)) * PV_dict;
    dict_pv_full(:,:,bb) = dict_pv_bb ;
end

% dict_pv_full = reshape(dict_pv_full,[num_acqs*turbo_factor,length(b1_val),size(dict_pv_full,2)]);

for cc = 1:size(dict_pv_full,2)
    for bb = 1:length(b1_val)
        dict_pv_full(:,cc,bb) = dict_pv_full(:,cc,bb) / sum(abs(dict_pv_full(:,cc,bb)).^2)^0.5;
    end
end



% PV dictionary-based

img_contrasts = recs_sub;

PV1_map = zeros([size(img_contrasts,1),size(img_contrasts,2),size(img_contrasts,4)]);
PV2_map = zeros([size(img_contrasts,1),size(img_contrasts,2),size(img_contrasts,4)]);
PV3_map = zeros([size(img_contrasts,1),size(img_contrasts,2),size(img_contrasts,4)]);
PV4_map = zeros([size(img_contrasts,1),size(img_contrasts,2),size(img_contrasts,4)]);

% b1 = ones([240,208]);
b1 = b1_map;
for ss = 1:size(img_contrasts,4)
    tic
    fprintf('slice: %d/%d... ',ss,size(img_contrasts,4));
    img_contrasts_ = img_contrasts(:,:,:,ss);
    
    [PV1_map(:,:,ss),PV2_map(:,:,ss),PV3_map(:,:,ss),PV4_map(:,:,ss)] = ...
        dict_fit_qalas_sub_pv_4part(permute(img_contrasts_,[1,2,4,3]),permute(dict_pv_full,[2,1,3]),permute(PV_dict,[2,1]),b1);
    toc
end
fprintf('done\n')
% mask = ones([240,208]);

mask = squeeze(abs(coeff_sub(:,:,1))>0.007);
figure;imagesc(PV1_map(:,:).*mask,[0,0.3]);colormap jet;colorbar;
figure;imagesc((PV2_map(:,:)+PV3_map(:,:)).*mask,[0.3,1]);colormap jet;colorbar;
% figure;imagesc(PV3_map(:,:).*mask,[0,1]);colormap jet;colorbar;
figure;imagesc(PV4_map(:,:).*mask,[0,0.6]);colormap jet;colorbar;
% 
% save('/autofs/cluster/berkin/shizhuo/result_invivo/mwf_0331/3d/3d_maps_x.mat','PV1_map','PV2_map','PV3_map','PV4_map','t1estimates_sub','t2estimates_sub')
% save('/autofs/cluster/berkin/shizhuo/result_invivo/mwf_0227/mwf/conv6_mwf_inv.mat','PV1_map','PV2_map','PV3_map')
% save('/autofs/cluster/berkin/shizhuo/result_invivo/mwf_0331/3d/opt5_3dz_paper.mat','t1estimates_sub','t2estimates_sub','PV1_map','PV2_map','PV3_map','PV4_map')
save('/autofs/space/marduk_001/users/shizhuo/Results/invivo/0710/Omni_R5_3dx.mat','t1estimates_sub','t2estimates_sub','PV1_map','PV2_map','PV3_map','PV4_map')
% save('/autofs/space/marduk_001/users/shizhuo/Results/invivo/0710/qalas_map_R5_mwfinvb1.mat','PV1_map','PV2_map','PV3_map','PV4_map')






