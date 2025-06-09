import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.autograd.forward_ad as fwad
import math


def safe_negate(tensor):
    return tensor * -1

# joint
def simulate_mwf(fa, gap, prep, parameters, num_acqs, tf, esp):
    alpha = torch.squeeze(fa).to(torch.device('cpu'))
    pure_gap = torch.squeeze(gap).to(torch.device('cpu'))
    prep = torch.squeeze(prep).to(torch.device('cpu'))
    num_reps = 5
    b1_val = torch.tensor([1])
    inv_eff = torch.tensor([1])
    etl = tf * esp

    # -sequence timings based on parameters defined above
    TE1 = prep[0]
    TE2 = prep[1]
    delT_M0_M1_1 = pure_gap[0]
    delT_M0_M1_2 = pure_gap[1]
    delT_M2_M3 = etl
    delT_M4_M5 = 12.8e-3
    delT_M5_M6 = 100e-3 - 6.45e-3
    delT_M3_M4 = pure_gap[2]
    delT_M6_M7 = etl
    delT_M7_M8 = pure_gap[3]
    delT_M8_M9 = etl
    if num_acqs > 4:
        delT_M9_M10 = pure_gap[4]
        delT_M10_M11 = etl
    if  num_acqs > 5:
        delT_M11_M12 = pure_gap[5]
        delT_M12_M13 = etl
    # delT_M13_end = 53.5e-3
    # -time between end of t2 prep pulse and first acquisition
    time_t2_prep_after = torch.tensor([9.7e-3])

    #####################################################################
    M0 = torch.tensor([1])
    Mz = torch.tensor([M0])

    Mxy_all = torch.zeros((num_acqs * tf, num_reps))

    for reps in range(num_reps):

        ech_ctr = 0
        acq_ctr = 0

        # ACQ0

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M0_M1_1 / parameters[1])
            Mz = Mz * (torch.sin(b1_val * torch.pi / 2) ** 2 * torch.exp(
                -(TE1 - time_t2_prep_after) / parameters[0]) + \
                torch.cos(b1_val * torch.pi / 2) ** 2 * torch.exp(
                -(TE1 - time_t2_prep_after) / parameters[1]))

            for q in range(tf):
                if q == 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-time_t2_prep_after / parameters[1])
                else:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        # ACQ1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M0_M1_2 / parameters[1])
            Mz = Mz * (torch.sin(b1_val * torch.pi / 2) ** 2 * torch.exp(
                -(TE2 - time_t2_prep_after) / parameters[0]) + \
                torch.cos(b1_val * torch.pi / 2) ** 2 * torch.exp(
                -(TE2 - time_t2_prep_after) / parameters[1]))
            for q in range(tf):
                if q == 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-time_t2_prep_after / parameters[1])
                else:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M3_M4 / parameters[1])
            # Mz = safe_negate(Mz)
            Mz = -Mz
            Mz = Mz * inv_eff
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M5_M6 / parameters[1])

            # ACQ2
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M7_M8 / parameters[1])

            # ACQ3
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M9_M10 / parameters[1])

            # ACQ4
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1
            acq_ctr = acq_ctr + 1

        if (acq_ctr < num_acqs):
            Mz = M0 - (M0 - Mz) * torch.exp(-delT_M11_M12 / parameters[1])

            # ACQ5
            for q in range(tf):
                if q > 0:
                    Mz = M0 - (M0 - Mz) * torch.exp(-esp / parameters[1])

                Mxy_all[ech_ctr, reps] = torch.sin(alpha[ech_ctr]) * Mz

                Mz = torch.cos(alpha[ech_ctr]) * Mz

                ech_ctr = ech_ctr + 1

        # Mz = M0 - (M0 - Mz) * torch.exp(-delT_M13_end / parameters[1])

    result = Mxy_all[:, -1] * parameters[2]
    result = result.to(torch.device('cuda'))
    return result


