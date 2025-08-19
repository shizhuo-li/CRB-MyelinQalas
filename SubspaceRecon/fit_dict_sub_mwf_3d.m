function [T1_map,T2_map,PD_map,IE_map] = ...
    fit_dict_sub_mwf_3d(imgs,dict,t1t2_lut_prune,estimate_pd_map,b1,inv_eff, ...
        TR, alpha_train, esp, turbo_factor, num_reps, echo2use, gap_between_readouts, time2relax_at_the_end, num_acqs,TE)

img         = abs(imgs);
N           = [size(img,1),size(img,2),size(img,3)];

% length_dict = length(dict);
length_dict = length(t1t2_lut_prune);

msk         = zeross(N);

for slc_select = 1:N(3)
    thresh = rsos(img(:,:,slc_select,:),4);
    thresh = mean(thresh(:)).*0.5;
    msk(:,:,slc_select) = imerode(imfill(rsos(img(:,:,slc_select,:),4) > thresh, 'holes'), strel('disk', 3));
end

% w/o msk
msk         = oness(N); 

estimate_pd_map = 0;
% b1
if length(b1) == 1
    b1      = ones(size(msk));
end

thre_high   = 1.35;
thre_low    = 0.65;
temp        = b1 .* msk;
temp(temp > thre_high)  = thre_high;
temp(temp < thre_low)   = thre_low;
img_b1      = temp .* msk;

num_b1_bins = size(dict,3);
% b1_val      = linspace(min(img_b1(msk==1)), max(img_b1(msk==1)), num_b1_bins);
b1_val      = 0.65:0.05:1.35;
% b1_val      = 1; % no b1 correction

if length(b1_val) == 1
    msk_b1 = msk;
else
    msk_b1 = zeross([N,length(b1_val)]);
    
    for t = 1:length(b1_val)
        if t > 1
            msk_b1(:,:,:,t)   = (img_b1 <= b1_val(t)) .* (img_b1 > b1_val(t-1));
        else
            msk_b1(:,:,:,t)   = msk.*(img_b1 <= b1_val(t));
        end
        
        if t == length(b1_val)
            msk_b1(:,:,:,t)   = img_b1 > b1_val(t-1);
        end
        
        msk_b1(:,:,:,t)       = msk_b1(:,:,:,t) .* msk;
    end
end

% img_b1  = ones(size(msk));
% inv_eff = 1;
% b1_val  = 1;
% msk_b1  = msk;

T1_map = zeross(N);
T2_map = zeross(N);
PD_map = zeross(N);
IE_map = zeross(N);

for slc_select = 1:N(3)
    
    for b = 1:length(b1_val)
        msk_slc = msk_b1(:,:,slc_select,b);
        num_vox = sum(msk_slc(:)~=0);
        
        if num_vox > 0
            img_slc = zeross([size(dict,2), num_vox]);
            
            for t = 1:size(dict,2)
                temp            = squeeze(img(:,:,slc_select,t));
                img_slc(t,:)    = temp(msk_slc~=0);
            end
            
            % res             = dict(:,:,b) * img_slc;
            % [~, max_idx]    = max(abs(res), [], 1);
            [~, max_idx]    = max(abs(dict(:,:,b) * img_slc), [], 1);
            
            max_idx_t1t2    = mod(max_idx, length_dict);
            max_idx_t1t2(max_idx_t1t2==0) = length_dict;
            
            res_map         = t1t2_lut_prune(max_idx_t1t2,:);
            max_idx_ie      = 1 + (max_idx - max_idx_t1t2) / length_dict;
            ie_to_use       = inv_eff(max_idx_ie);
            % ie_to_use = inv_eff(max_idx_t1t2,1);
            
            if estimate_pd_map
                % [Mz_sim, Mxy_sim] = sim_qalas_pd_b1_eff_T2_v0_YH(TR, alpha_deg, esp, ...
                %                         turbo_factor, res_map(:,1)*1e-3, res_map(:,2)*1e-3, ...
                %                         num_reps, echo2use, gap_between_readouts, ...
                %                         time2relax_at_the_end, b1_val(b), ie_to_use.');
                [~,Mxy_sim] = sim_qalas_acqs_mwf_invrange(TR, alpha_train, esp, turbo_factor, res_map(:,1)*1e-3,...
                           res_map(:,2)*1e-3, num_reps, num_acqs, gap_between_readouts, time2relax_at_the_end,...
                           b1_val(b),ie_to_use,TE);
            end
            
            t1_map = zeross(N(1:2));
            t1_map(msk_slc==1) = res_map(:,1);
            
            t2_map = zeross(N(1:2));
            t2_map(msk_slc==1) = res_map(:,2);
            
            ie_map = zeross(N(1:2));
            ie_map(msk_slc==1) = ie_to_use;
            
            
            if estimate_pd_map
                Mxy_sim_use = abs(Mxy_sim(:,:,end));
                scl         = zeross([num_vox,1]);
                
                for idx = 1:size(Mxy_sim_use,2)
                    scl(idx) = Mxy_sim_use(:,idx) \ img_slc(:,idx);
                end
                
                pd_map                  = zeross(N(1:2));
                pd_map(msk_slc~=0)      = scl;
                PD_map(:,:,slc_select) = PD_map(:,:,slc_select) + pd_map;
            end
            
            T1_map(:,:,slc_select) = T1_map(:,:,slc_select) + t1_map;
            T2_map(:,:,slc_select) = T2_map(:,:,slc_select) + t2_map;
            IE_map(:,:,slc_select) = IE_map(:,:,slc_select) + ie_map;
        end
    end
    % toc
end

end