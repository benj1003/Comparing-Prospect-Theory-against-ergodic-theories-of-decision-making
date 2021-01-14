import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import beta
import pandas as pd
import datetime
 
print("Time:")
print(datetime.datetime.now())
print()
####  MISSINGS
# - Plot beta params (and hyperparams) for all models

#Plots
plot_Rvalues            = False
R_table                 = False

plot_ModelIndicator     = False
redundancy_plots        = False

plot_tw_params          = False
plot_tw_eta             = False
plot_tw_eta_hypers      = False
plot_beta_tw            = False
plot_beta_tw_hyper      = False

plot_pt_org_params      = False
plot_alphaGain_p        = False
plot_alphaGain_p_hyper  = False
plot_alphaLoss_p        = False
plot_alphaLoss_p_hyper  = False
plot_lambda_p           = False
plot_lambda_p_hyper     = False
plot_beta_p             = False
plot_beta_p_hyper       = False

plot_pt_gain_params     = False
plot_Wvalues            = False
plot_weighdist          = False
plot_weighthyper        = False
plot_alphaGain_pg       = False
plot_alphaGain_pg_hyper = False
plot_beta_pg            = False
plot_beta_pg_hyper      = False

plot_iso_params         = True
plot_iso_eta            = False
plot_iso_eta_hypers     = False
plot_beta_iso           = True
plot_beta_iso_hyper     = False

fig_nr = 1
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

##Flat prior
# file_name = 'JAGS_StrongModels_Subjectwise_ExpandZ_pt_gain_FlatZprior_allData_burn_10000_samps_5000_chains_4_05-04-2020 21-51' 

##PT_gain
# file_name = 'JAGS_test_PT_PG_v4_FlatZprior_allData_burn_10000_samps_5000_chains_4_04-30-2020 05-23'

##PT_original
# file_name = 'JAGS_test_PT_PG_v4_PT_prior_1_allData_burn_10000_samps_5000_chains_4_05-01-2020 04-26'

##TW
# file_name = 'JAGS_StrongModels_Subjectwise_ExpandZ_pt_gain_TW_prior_1_allData_burn_10000_samps_5000_chains_4_05-07-2020 19-51'

##ISO
file_name = 'JAGS_StrongModels_Subjectwise_ExpandZ_pt_gain_ISO_prior_1_allData_burn_10000_samps_5000_chains_4_05-09-2020 09-23'

with h5py.File(f'samples_stats/{file_name}.mat', 'r') as file:
    print(list(file.keys()))

    # print("\n SAMPLES:")
    ##SAMPLES
    # print(list(file['samples'].keys()))
    #Modelindicator
    z = file['samples'].get('z').value

    # # Time optimal
    # beta_tw                 = file['samples'].get('beta_tw').value
    # eta_tw                  = file['samples'].get('eta_tw').value

    # mu_eta_tw               = file['samples'].get('mu_eta_tw').value
    # sigma_eta_tw            = file['samples'].get('sigma_eta_tw').value
    # tau_eta_tw              = file['samples'].get('tau_eta_tw').value
    # mu_log_beta_tw          = file['samples'].get('mu_log_beta_tw').value
    # sigma_log_beta_tw       = file['samples'].get('sigma_log_beta_tw').value
    # tau_log_beta_tw         = file['samples'].get('tau_log_beta_tw').value

    #PT_original
    # alphaGain_p             = file['samples'].get('alphaGain_p').value
    # alphaLoss_p             = file['samples'].get('alphaLoss_p').value
    # lambda_p                = file['samples'].get('lambda_p').value
    # beta_p                  = file['samples'].get('beta_p').value
    
    # mu_log_alphaGain_p      = file['samples'].get('mu_log_alphaGain_p').value
    # sigma_log_alphaGain_p   = file['samples'].get('sigma_log_alphaGain_p').value
    # tau_log_alphaGain_p     = file['samples'].get('tau_log_alphaGain_p').value
    # mu_log_alphaLoss_p      = file['samples'].get('mu_log_alphaLoss_p').value
    # sigma_log_alphaLoss_p   = file['samples'].get('sigma_log_alphaLoss_p').value
    # tau_log_alphaLoss_p     = file['samples'].get('tau_log_alphaLoss_p').value
    # mu_log_lambda_p         = file['samples'].get('mu_log_lambda_p').value
    # sigma_log_lambda_p      = file['samples'].get('sigma_log_lambda_p').value
    # tau_log_lambda_p        = file['samples'].get('tau_log_lambda_p').value
    # mu_log_beta_p           = file['samples'].get('mu_log_beta_p').value
    # sigma_log_beta_p        = file['samples'].get('sigma_log_beta_p').value
    # tau_log_beta_p          = file['samples'].get('tau_log_beta_p').value


    # #PT_gain
    # alphaGain_pg            = file['samples'].get('alphaGain_pg').value
    # beta_pg                 = file['samples'].get('beta_pg').value
    # w_pg                    = file['samples'].get('w_pg').value
    # mu_log_alphaGain_pg     = file['samples'].get('mu_log_alphaGain_pg').value
    # sigma_log_alphaGain_pg  = file['samples'].get('sigma_log_alphaGain_pg').value
    # tau_log_alphaGain_pg    = file['samples'].get('tau_log_alphaGain_pg').value
    # mu_log_beta_pg          = file['samples'].get('mu_log_beta_pg').value
    # sigma_log_beta_pg       = file['samples'].get('sigma_log_beta_pg').value
    # tau_log_beta_pg         = file['samples'].get('tau_log_beta_pg').value
    # weight_a_pg             = file['samples'].get('weight_a_pg').value
    # weight_b_pg             = file['samples'].get('weight_b_pg').value

    #Iso
    beta_iso                = file['samples'].get('beta_iso').value
    eta_iso                 = file['samples'].get('eta_iso').value

    mu_eta_iso              = file['samples'].get('mu_eta_iso').value
    sigma_eta_iso           = file['samples'].get('sigma_eta_iso').value
    tau_eta_iso             = file['samples'].get('tau_eta_iso').value
    mu_log_beta_iso         = file['samples'].get('mu_log_beta_iso').value
    sigma_log_beta_iso      = file['samples'].get('sigma_log_beta_iso').value
    tau_log_beta_iso        = file['samples'].get('tau_log_beta_iso').value

    # print("\n R_hat values")
    #Convergens test (only on priors)
    # print(list(file['stats']['Rhat'].keys()))
    # R_z               = file['stats']['Rhat'].get('z').value
    # R_beta_tw         = file['stats']['Rhat'].get('beta_tw').value
    # R_eta_tw          = file['stats']['Rhat'].get('eta_tw').value
    # R_alphaGain       = file['stats']['Rhat'].get('alphaGain').value
    # R_alphaLoss       = file['stats']['Rhat'].get('alphaLoss').value
    # R_lambda_pt       = file['stats']['Rhat'].get('lambda').value
    # R_beta_pt         = file['stats']['Rhat'].get('beta_pt').value
    # R_alphaGain_pg    = file['stats']['Rhat'].get('alphaGain_pg').value
    # R_beta_pg         = file['stats']['Rhat'].get('beta_pg').value
    # R_wpg             = file['stats']['Rhat'].get('w_pg').value
    # R_beta_iso        = file['stats']['Rhat'].get('beta_iso').value
    # R_eta_iso         = file['stats']['Rhat'].get('eta_iso').value

print()
tw_model  = [1,5,9,13]
pt_model  = [2,6,10,14]
pg_model  = [3,7,11,15]
iso_model = [4,8,12,16]

# pt_model  = [2,4,6,8]
# tw_model  = [0,0,0,0]
# pg_model  = [1,3,5,7]
# iso_model = [0,0,0,0]

# pt_model  = [0,0,0,0]
# tw_model  = [2,6,10,14]
# pg_model  = [1,3,5,7]
# iso_model = [0,0,0,0]

n_subjects = np.shape(z)[0]
n_samples = np.shape(z)[1]
n_trials = np.shape(z)[2]

print(f"Number of subjects = {n_subjects}")
print(f"Number of trials = {n_trials}")
print(f"Number of samples = {n_samples}\n")

####### Convergens tests
if plot_Rvalues:
    print("\n Convergence test")
    max_r_value = 1.1
    #Modelindicator
    plt.figure(fig_nr)
    fig_nr += 1
    plt.title("R_hat values for z")
    plt.plot(range(n_subjects), R_z[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')

    #Time optimal
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("R_hat for Priors in Time optimal")
    plt.subplot(211)
    plt.title("Beta")
    plt.plot(range(n_subjects), R_beta_tw[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(212)
    plt.title("Eta")
    plt.plot(range(n_subjects), R_eta_tw[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')

    #PT_original
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("R_hat for Priors in PT_original")
    plt.subplot(221)
    plt.title("AlphaGain")
    plt.plot(range(n_subjects), R_alphaGain[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(222)
    plt.title("AlphaLoss")
    plt.plot(range(n_subjects), R_alphaLoss[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(223)
    plt.title("Lambda")
    plt.plot(range(n_subjects), R_lambda_pt[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(224)
    plt.title("Beta")
    plt.plot(range(n_subjects), R_beta_pt[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')

    #PT_gain
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("R_hat for Priors in PT_gain")
    plt.subplot(221)
    plt.title("AlphaGain")
    plt.plot(range(n_subjects), R_alphaGain_pg[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(222)
    plt.title("Beta")
    plt.plot(range(n_subjects), R_beta_pg[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(212)
    plt.title("Weights")
    plt.plot(range(n_subjects), R_wpg[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')

    #Iso
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("R_hat for Priors in Iso")
    plt.subplot(211)
    plt.title("Beta")
    plt.plot(range(n_subjects), R_beta_iso[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')
    plt.subplot(212)
    plt.title("Eta")
    plt.plot(range(n_subjects), R_eta_iso[0], 'o')
    plt.xlim(-1,18)
    plt.axhline(y=max_r_value, color='r', linestyle='--')

    plt.show()

if R_table:
    subjects = [a + 1 for a in range(18)] 
    df = pd.DataFrame({'Subject': subjects, 'Z': R_z[0], 'alpha_gain': R_alphaGain[0], 'alpha_loss': R_alphaLoss[0], 'Lambda': R_lambda_pt[0], 'beta_add_pt': R_beta_pt[0],'beta_mul_pt': R_beta_pt[1], \
        'alpha': R_alphaGain_pg[0], 'Weights': R_wpg[0], 'beta_add_pg': R_beta_pg[0],'beta_mul_pg': R_beta_pg[1]})
    df = df.round(4)
    print(df.to_latex(index=False))
    df = pd.DataFrame({'Subject': subjects,  'eta': R_eta_iso[0], 'beta_add_iso': R_beta_iso[0],'beta_mul_iso': R_beta_iso[1], \
                            'eta_add': R_eta_tw[0],'eta_mul': R_eta_tw[1], 'beta_add_tw': R_beta_tw[0],'beta_mul_tw': R_beta_tw[1]})
    df = df.round(4)
    print(df.to_latex(index=False))

####### Modelindicator
if plot_ModelIndicator:
    z_sample_choices_extended_subjects = np.empty([n_samples*n_trials , n_subjects])
    z_sample_choices_extended_subjects_helper = []
    z_sample_choices_subjects = []
    z_sample_choices_extended = []
    z_sample_choices = []
    z_sample_choices_trials = []

    TW = 0
    PT = 0
    PG = 0
    ISO = 0
    tmp_extra_2 = []
    for i in range(n_subjects): 
        tmp =  []
        tmp_extra = []
        TW_tmp = 0
        PT_tmp = 0
        PG_tmp = 0
        ISO_tmp = 0
        for l in range(n_trials):
            TW_tmp_extra = 0
            PT_tmp_extra = 0
            PG_tmp_extra = 0
            ISO_tmp_extra = 0
            for j in range(n_samples):
                #Append for extended
                tmp.append(z[i,j,l])

                z_sample_choices_extended.append(z[i,j,l])

                #Condence choices
                if z[i,j,l] in tw_model:
                    TW += 1
                    TW_tmp += 1
                    TW_tmp_extra += 1
                elif z[i,j,l] in pt_model:
                    PT += 1
                    PT_tmp += 1
                    PT_tmp_extra += 1
                elif z[i,j,l] in pg_model:
                    PG += 1
                    PG_tmp += 1
                    PG_tmp_extra += 1
                elif z[i,j,l] in iso_model:
                    ISO += 1
                    ISO_tmp += 1
                    ISO_tmp_extra += 1
                else:
                    print("\n\n\nWARNING: Something is wrong!!!\n\n\n")
            tmp_extra.append([TW_tmp_extra/n_samples,PT_tmp_extra/n_samples, PG_tmp_extra/n_samples,ISO_tmp_extra/n_samples])
        z_sample_choices_extended_subjects[:,i] = tmp
        z_sample_choices_extended_subjects_helper.append(Counter(tmp))
        z_sample_choices_subjects.append([TW_tmp/(n_samples*n_trials),PT_tmp/(n_samples*n_trials), PG_tmp/(n_samples*n_trials),ISO_tmp/(n_samples*n_trials)])
        tmp_extra_2.append(tmp_extra)
    z_sample_choices.append([TW/(n_samples*n_trials*n_subjects),PT/(n_samples*n_trials*n_subjects),PG/(n_samples*n_trials*n_subjects),ISO/(n_samples*n_trials*n_subjects)])
    z_sample_choices_trials.append(tmp_extra_2)

    x = ['TW','PT', 'PG','ISO']
    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Model indicator for each Subject")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title("subject %.0f - Rhat = %.2f" %(j+1, R_z[0][j]), )
        plt.bar(x, z_sample_choices_subjects[j])
        plt.ylim([0,1])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Model indicator for each Subject (extended)")
    for j in range(n_subjects):
        plt.subplot(6,3,j+1)
        plt.title(f"subject {j+1}")
        # print(f"\nSubject {j+1} has the following: {z_sample_choices_extended_subjects_helper[j]}")
        plt.hist(z_sample_choices_extended_subjects[:,j], bins = 16)
        plt.xlim([1,16])
        # plt.ylim([0,1])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

    plt.figure(fig_nr)
    fig_nr += 1
    plt.suptitle("Modelindicator")
    plt.bar(x,z_sample_choices[0])
    plt.ylim([0,1])

    plt.figure(fig_nr)
    fig_nr += 1
    plt.title("Modelindicator (extended)")
    plt.hist(z_sample_choices_extended)
    # plt.ylim([0,1])

    plt.show()
    
    for i in range(n_subjects):
        plt.figure(fig_nr)
        fig_nr += 1
        for l in range(n_trials):
            plt.suptitle(f"Subject n. {i+1}")
            plt.subplot(4,1,l+1)
            plt.title(f"Trial {l+1}")
            plt.bar(x,z_sample_choices_trials[0][i][l])
            plt.ylim([0,1])
    # plt.show()

    print(z_sample_choices_subjects)

if redundancy_plots:
    x = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
    #REDUNDANCY PLOT
    z_sample_choices_trials = []
    tmp_extra_2 = []
    for i in range(n_subjects): 
        tmp_extra = []
        for l in range(n_trials):
            TW_tmp_1 = 0
            TW_tmp_2 = 0
            TW_tmp_3 = 0
            TW_tmp_4 = 0
            PT_tmp_1 = 0
            PT_tmp_2 = 0
            PT_tmp_3 = 0
            PT_tmp_4 = 0
            PG_tmp_1 = 0
            PG_tmp_2 = 0
            PG_tmp_3 = 0
            PG_tmp_4 = 0
            ISO_tmp_1 = 0
            ISO_tmp_2 = 0
            ISO_tmp_3 = 0
            ISO_tmp_4 = 0
            for j in range(n_samples):
                if z[i,j,l] == 1:
                    TW_tmp_1 += 1
                elif z[i,j,l] == 5:
                    TW_tmp_2 += 1
                elif z[i,j,l] == 9:
                    TW_tmp_3 += 1
                elif z[i,j,l] == 13:
                    TW_tmp_4 += 1
                elif z[i,j,l] == 2:
                    PT_tmp_1 += 1
                elif z[i,j,l] == 6:
                    PT_tmp_2 += 1
                elif z[i,j,l] == 10:
                    PT_tmp_3 += 1
                elif z[i,j,l] == 14:
                    PT_tmp_4 += 1
                elif z[i,j,l] == 3:
                    PG_tmp_1 += 1
                elif z[i,j,l] == 7:
                    PG_tmp_2 += 1
                elif z[i,j,l] == 11:
                    PG_tmp_3 += 1
                elif z[i,j,l] == 15:
                    PG_tmp_4 += 1
                elif z[i,j,l] == 4:
                    ISO_tmp_1 += 1
                elif z[i,j,l] == 8:
                    ISO_tmp_2 += 1
                elif z[i,j,l] == 12:
                    ISO_tmp_3 += 1
                elif z[i,j,l] == 16:
                    ISO_tmp_4 += 1

            tmp_extra.append([TW_tmp_1/n_samples,PT_tmp_1/n_samples, PG_tmp_1/n_samples,ISO_tmp_1/n_samples,TW_tmp_2/n_samples,PT_tmp_2/n_samples, PG_tmp_2/n_samples,ISO_tmp_2/n_samples,TW_tmp_3/n_samples,PT_tmp_3/n_samples, PG_tmp_3/n_samples,ISO_tmp_3/n_samples,TW_tmp_4/n_samples,PT_tmp_4/n_samples, PG_tmp_4/n_samples,ISO_tmp_4/n_samples])
        tmp_extra_2.append(tmp_extra)
    z_sample_choices_trials.append(tmp_extra_2)

    for i in range(n_subjects):
        plt.figure(fig_nr)
        fig_nr += 1
        for l in range(n_trials):
            plt.suptitle(f"Subject n. {i+1}")
            plt.subplot(4,1,l+1)
            # plt.title(f"Trial {l+1}")
            plt.bar(x,z_sample_choices_trials[0][i][l])
            plt.ylim([0,1])
            plt.yticks([0,1])
            if l != 3:
                plt.xticks([], [])
                # plt.xlabel([' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '])
    plt.show()

####### Time optimal
if plot_tw_params:
    if plot_tw_eta:
        eta_tw_add_all = []
        eta_tw_mul_all = []
        eta_tw_add_ = []
        eta_tw_mul_ = []
        for i in range(n_subjects):
            tmp_add = []
            tmp_mul = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in tw_model:
                        eta_tw_add_all.append(eta_tw[0,i,j,l])
                        eta_tw_mul_all.append(eta_tw[1,i,j,l])
                        tmp_add.append(eta_tw[0,i,j,l])
                        tmp_mul.append(eta_tw[1,i,j,l])
            eta_tw_add_.append(tmp_add)
            eta_tw_mul_.append(tmp_mul)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for $\eta$")
        for j in range(n_subjects):
            plt.subplot(6,3,j+1)
            plt.title(f"Subject {j+1}")
            plt.hist(eta_tw_add_[j], bins = 50, density = True)
            plt.hist(eta_tw_mul_[j], color="red", bins = 50, density = True)
            plt.xticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
            plt.xlim([-6,6])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\eta$")
        plt.xticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
        plt.xlim([-6,6])
        sns.kdeplot(eta_tw_add_all, label = 'Additive')
        sns.kdeplot(eta_tw_mul_all, color = 'r', label = 'Multiplicative')
        plt.legend(loc = 'upper left')
        plt.show()

    if plot_tw_eta_hypers:
        eta_tw_tau = []
        eta_tw_sigma = []
        for l in range(n_trials):
            for j in range(n_samples):
                eta_tw_tau.append(tau_eta_tw[j,l])
                eta_tw_sigma.append(sigma_eta_tw[j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\eta$")
        plt.subplot(211)
        plt.title(f"Distribution for $\\tau$")
        sns.kdeplot(eta_tw_tau)
        plt.xlim([-200,500])
        plt.subplot(212)
        plt.title(f"Distribution for $\sigma$")
        sns.kdeplot(eta_tw_sigma)
        plt.show()
    if plot_beta_tw:
        beta_tw_all_add = []
        beta_tw_means_add = []
        beta_tw_add_ = []
        beta_tw_all_mul = []
        beta_tw_means_mul = []
        beta_tw_mul_ = []
        for i in range(n_subjects):
            tmp_add = []
            tmp_mul = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in tw_model:
                        beta_tw_all_add.append(beta_tw[0,i,j,l])
                        tmp_add.append(beta_tw[0,i,j,l])
                        beta_tw_all_mul.append(beta_tw[1,i,j,l])
                        tmp_mul.append(beta_tw[1,i,j,l])
            beta_tw_means_add.append(np.mean(tmp_add))
            beta_tw_add_.append(tmp_add)
            beta_tw_means_mul.append(np.mean(tmp_mul))
            beta_tw_mul_.append(tmp_mul)

        fig = plt.figure(fig_nr)
        fig_nr += 1
        outer = gridspec.GridSpec(6, 3, wspace=0.2, hspace=1)

        for j in range(n_subjects):
            inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                            subplot_spec=outer[j], wspace=0.2, hspace=0.4)
            
            # set outer titles
            ax = plt.Subplot(fig, outer[j])
            ax.set_title("Subject {}".format(j+1),  y=1.2)
            ax.axis('off')
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[0])
            ax.set_title(f"Additive")
            ax.set_xlim([-2,50])
            t = ax.hist(beta_tw_add_[j], bins=np.arange(min(beta_tw_add_[j]), max(beta_tw_add_[j]) + 0.5, 0.5), density = True)
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[1])
            t = ax.hist(beta_tw_mul_[j], bins=np.arange(min(beta_tw_mul_[j]), max(beta_tw_mul_[j]) + 0.5, 0.5), density = True)
            ax.set_title(f"Multiplicative")
            ax.set_xlim([-2,50])
            fig.add_subplot(ax)
        fig.suptitle('Distributions of $\\beta$')

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\\beta$")
        plt.subplot(121)
        plt.title("Additive")
        plt.hist(beta_tw_all_add, bins=np.arange(min(beta_tw_all_add), max(beta_tw_all_add) + 0.5, 0.5), density = True)
        # sns.kdeplot(beta_tw_all_add)
        plt.xlim([-2,50])
        plt.subplot(122)
        plt.title("Multiplikative")
        plt.hist(beta_tw_all_mul, bins=np.arange(min(beta_tw_all_mul), max(beta_tw_all_mul) + 0.5, 0.5), density = True)
        # sns.kdeplot(beta_tw_all_mul)
        plt.xlim([-2,50])
        plt.show()

    if plot_beta_tw_hyper:
        beta_tw_mu_add = []
        beta_tw_mu_mul = []
        beta_tw_tau_add = []
        beta_tw_tau_mul = []
        beta_tw_sigma_add = []
        beta_tw_sigma_mul = []
        for l in range(n_trials):
            for j in range(n_samples):
                beta_tw_mu_add.append(np.exp(mu_log_beta_tw[0,j,l]))
                beta_tw_mu_mul.append(np.exp(mu_log_beta_tw[1,j,l]))
                beta_tw_tau_add.append(tau_log_beta_tw[0,j,l])
                beta_tw_tau_mul.append(tau_log_beta_tw[1,j,l])
                beta_tw_sigma_add.append(sigma_log_beta_tw[0,j,l])
                beta_tw_sigma_mul.append(sigma_log_beta_tw[1,j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\\beta$")
        plt.subplot(221)
        plt.title(f"Distribution for $\mu_a$")
        sns.kdeplot(beta_tw_mu_add)
        plt.subplot(222)
        plt.title(f"Distribution for $\mu_m$")
        sns.kdeplot(beta_tw_mu_mul)
        plt.subplot(245)
        plt.title(f"Distribution for $\\tau_a$")
        sns.kdeplot(beta_tw_tau_add)
        plt.xlim([-200,500])
        plt.subplot(246)
        plt.title(f"Distribution for $\\tau_m$")
        sns.kdeplot(beta_tw_tau_mul)
        plt.xlim([-200,500])
        plt.subplot(247)
        plt.title(f"Distribution for $\sigma_a$")
        sns.kdeplot(beta_tw_sigma_add)
        plt.subplot(248)
        plt.title(f"Distribution for $\sigma_m$")
        sns.kdeplot(beta_tw_sigma_mul)
        plt.show()



####### PT_original
if plot_pt_org_params:
    print("\n PT_original Parameters")
    if plot_alphaGain_p:
        alphaGain_p_all = []
        alphaGain_p_means = []
        alphaGain_p_ = []
        for i in range(n_subjects):
            tmp = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in pt_model:
                        alphaGain_p_all.append(alphaGain_p[i,j,l])
                        tmp.append(alphaGain_p[i,j,l])
            alphaGain_p_means.append(np.mean(tmp))
            alphaGain_p_.append(tmp)

        print(alphaGain_p_means)
        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for $\\alpha_G$")
        for j in range(n_subjects):
            plt.subplot(6,3,j+1)
            plt.title(f"Subject {j+1}")
            plt.hist(alphaGain_p_[j], density = True, bins = 100)
            # plt.axvline(np.mean(alphaGain_p_[j]),color='red')
            plt.xlim([0,2])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\\alpha_G$")
        sns.kdeplot(alphaGain_p_all)
        # plt.xlim([0,1])
        # [plt.axvline(_w, linewidth=1, color='red') for _w in alphaGain_p_means]
        plt.show()

    if plot_alphaGain_p_hyper:
        alphaGain_p_mu = []
        alphaGain_p_tau = []
        alphaGain_p_sigma = []
        for l in range(n_trials):
            for j in range(n_samples):
                alphaGain_p_mu.append(np.exp(mu_log_alphaGain_p[j,l]))
                alphaGain_p_tau.append(tau_log_alphaGain_p[j,l])
                alphaGain_p_sigma.append(sigma_log_alphaGain_p[j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\\alpha_G$")
        plt.subplot(211)
        plt.title(f"Distribution for $\mu$")
        sns.kdeplot(alphaGain_p_mu)
        plt.subplot(223)
        plt.title(f"Distribution for $\\tau$")
        sns.kdeplot(alphaGain_p_tau)
        plt.xlim(right=200)
        plt.subplot(224)
        plt.title(f"Distribution for $\sigma$")
        sns.kdeplot(alphaGain_p_sigma)
        plt.show()
        
    if plot_alphaLoss_p:
        alphaLoss_p_all = []
        alphaLoss_p_means = []
        alphaLoss_p_ = []
        for i in range(n_subjects):
            tmp = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in pt_model:
                        alphaLoss_p_all.append(alphaLoss_p[i,j,l])
                        tmp.append(alphaLoss_p[i,j,l])
            alphaLoss_p_means.append(np.mean(tmp))
            alphaLoss_p_.append(tmp)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distribution for $\\alpha_L$")
        for j in range(n_subjects):
            plt.subplot(6,3,j+1)
            plt.title(f"Subject {j+1}")
            plt.hist(alphaLoss_p_[j], bins = 100)
            # plt.axvline(np.mean(alphaLoss_p_[j]),color='red')
            plt.xlim([0,2])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\\alpha_L$")
        sns.kdeplot(alphaLoss_p_all)
        # plt.xlim([0,1])
        # [plt.axvline(_w, linewidth=1, color='red') for _w in alphaLoss_p_means]
        plt.show()

    if plot_alphaLoss_p_hyper:
        alphaLoss_p_mu = []
        alphaLoss_p_tau = []
        alphaLoss_p_sigma = []
        for l in range(n_trials):
            for j in range(n_samples):
                alphaLoss_p_mu.append(np.exp(mu_log_alphaLoss_p[j,l]))
                alphaLoss_p_tau.append(tau_log_alphaLoss_p[j,l])
                alphaLoss_p_sigma.append(sigma_log_alphaLoss_p[j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\\alpha_L$")
        plt.subplot(211)
        plt.title(f"Distribution for $\mu$")
        sns.kdeplot(alphaLoss_p_mu)
        plt.subplot(223)
        plt.title(f"Distribution for $\\tau$")
        sns.kdeplot(alphaLoss_p_tau)
        plt.xlim([-50,200])
        plt.subplot(224)
        plt.title(f"Distribution for $\sigma$")
        sns.kdeplot(alphaLoss_p_sigma)
        plt.show()

    if plot_lambda_p:
        lambda_p_all = []
        lambda_p_means = []
        lambda_p_ = []
        for i in range(n_subjects):
            tmp = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in pt_model:
                        lambda_p_all.append(lambda_p[i,j,l])
                        tmp.append(lambda_p[i,j,l])
            lambda_p_means.append(np.mean(tmp))
            lambda_p_.append(tmp)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for $\lambda$")
        for j in range(n_subjects):
            plt.subplot(6,3,j+1)
            plt.title(f"Subject {j+1}")
            plt.hist(lambda_p_[j], bins = 100)
            # plt.axvline(np.mean(lambda_p_[j]),color='red')
            plt.xlim([0,5])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\lambda$")
        sns.kdeplot(lambda_p_all)
        plt.xlim(right=5)
        # [plt.axvline(_w, linewidth=1, color='red') for _w in lambda_p_means]
        plt.show()

    if plot_lambda_p_hyper:
        lambda_p_mu = []
        lambda_p_tau = []
        lambda_p_sigma = []
        for l in range(n_trials):
            for j in range(n_samples):
                lambda_p_mu.append(np.exp(mu_log_lambda_p[j,l]))
                lambda_p_tau.append(tau_log_lambda_p[j,l])
                lambda_p_sigma.append(sigma_log_lambda_p[j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\lambda$")
        plt.subplot(211)
        plt.title(f"Distribution for $\mu$")
        sns.kdeplot(lambda_p_mu)
        plt.subplot(223)
        plt.title(f"Distribution for $\\tau$")
        sns.kdeplot(lambda_p_tau)
        plt.subplot(224)
        plt.title(f"Distribution for $\sigma$")
        sns.kdeplot(lambda_p_sigma)
        plt.show()

    if plot_beta_p:
        beta_p_all_add = []
        beta_p_means_add = []
        beta_p_add_ = []
        beta_p_all_mul = []
        beta_p_means_mul = []
        beta_p_mul_ = []
        for i in range(n_subjects):
            tmp_add = []
            tmp_mul = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in pt_model:
                        beta_p_all_add.append(beta_p[0,i,j,l])
                        tmp_add.append(beta_p[0,i,j,l])
                        beta_p_all_mul.append(beta_p[1,i,j,l])
                        tmp_mul.append(beta_p[1,i,j,l])
            beta_p_means_add.append(np.mean(tmp_add))
            beta_p_add_.append(tmp_add)
            beta_p_means_mul.append(np.mean(tmp_mul))
            beta_p_mul_.append(tmp_mul)

        # print(beta_p_add_)
        fig = plt.figure(fig_nr)
        fig_nr += 1
        outer = gridspec.GridSpec(6, 3, wspace=0.2, hspace=1)

        for j in range(n_subjects):
            inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                            subplot_spec=outer[j], wspace=0.2, hspace=0.4)
            
            # set outer titles
            ax = plt.Subplot(fig, outer[j])
            ax.set_title("Subject {}".format(j+1),  y=1.2)
            ax.axis('off')
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[0])
            ax.set_title(f"Additive")
            ax.set_xlim([-2,10])
            # print(beta_p_add_[j])
            t = ax.hist(beta_p_add_[j], bins=np.arange(min(beta_p_add_[j]), max(beta_p_add_[j]) + 0.1, 0.1), density = True)
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[1])
            t = ax.hist(beta_p_mul_[j], bins=np.arange(min(beta_p_mul_[j]), max(beta_p_mul_[j]) + 0.1, 0.1), density = True)
            ax.set_title(f"Multiplicative")
            ax.set_xlim([-2,10])
            fig.add_subplot(ax)
        fig.suptitle('Distributions of $\\beta$')

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\\beta$")
        plt.subplot(121)
        plt.title("Additive")
        plt.hist(beta_p_all_add, bins=np.arange(min(beta_p_all_add), max(beta_p_all_add) + 0.1, 0.1), density = True)
        # sns.kdeplot(beta_tw_all_add)
        plt.xlim([-2,10])
        plt.subplot(122)
        plt.title("Multiplikative")
        plt.hist(beta_p_all_mul, bins=np.arange(min(beta_p_all_mul), max(beta_p_all_mul) + 0.1, 0.1), density = True)
        # sns.kdeplot(beta_tw_all_mul)
        plt.xlim([-2,10])
        plt.show()

    if plot_beta_p_hyper:
        lambda_p_mu_add = []
        lambda_p_mu_mul = []
        lambda_p_tau_add = []
        lambda_p_tau_mul = []
        lambda_p_sigma_add = []
        lambda_p_sigma_mul = []
        for l in range(n_trials):
            for j in range(n_samples):
                lambda_p_mu_add.append(np.exp(mu_log_beta_p[0,j,l]))
                lambda_p_mu_mul.append(np.exp(mu_log_beta_p[1,j,l]))
                lambda_p_tau_add.append(sigma_log_beta_p[0,j,l])
                lambda_p_tau_mul.append(sigma_log_beta_p[1,j,l])
                lambda_p_sigma_add.append(tau_log_beta_p[0,j,l])
                lambda_p_sigma_mul.append(tau_log_beta_p[1,j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\\beta$")
        plt.subplot(221)
        plt.title(f"Distribution for $\mu_a$")
        sns.kdeplot(lambda_p_mu_add)
        plt.subplot(222)
        plt.title(f"Distribution for $\mu_m$")
        sns.kdeplot(lambda_p_mu_mul)
        plt.subplot(245)
        plt.title(f"Distribution for $\\tau_a$")
        sns.kdeplot(lambda_p_tau_add)
        plt.subplot(246)
        plt.title(f"Distribution for $\\tau_m$")
        sns.kdeplot(lambda_p_tau_mul)
        plt.subplot(247)
        plt.title(f"Distribution for $\sigma_a$")
        sns.kdeplot(lambda_p_sigma_add)
        plt.subplot(248)
        plt.title(f"Distribution for $\sigma_m$")
        sns.kdeplot(lambda_p_sigma_mul)
        plt.show()

####### PT_gain
if plot_pt_gain_params:
    print("\n PT_gain Parameters")
    if plot_Wvalues:
        ## plotting all the weights
        for i in range(n_subjects):
            plt.figure(fig_nr)
            fig_nr += 1
            plt.subplot(411)
            plt.title(f"Weights for subject {i+1}")
            plt.plot(range(n_samples), w_pg[i,:,0], color = "red")
            plt.subplot(412)
            plt.plot(range(n_samples), w_pg[i,:,1], color = "red")
            plt.subplot(413)
            plt.plot(range(n_samples), w_pg[i,:,2], color = "red")
            plt.subplot(414)
            plt.plot(range(n_samples), w_pg[i,:,3], color = "red")
        plt.show()

    if plot_weighdist:
        w_pg_all = []
        w_pg_means = []
        w_pg_ = []
        for i in range(n_subjects):
            tmp = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in pg_model:
                        w_pg_all.append(w_pg[i,j,l])
                        tmp.append(w_pg[i,j,l])
            w_pg_means.append(np.mean(tmp))
            w_pg_.append(tmp)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distribution on $w(p)$")
        for j in range(n_subjects):
            plt.subplot(6,3,j+1)
            plt.title(f"Subject {j+1}")
            plt.hist(w_pg_[j], density = True, bins = 100)
            # plt.axvline(np.mean(w_pg_[j]),color='red')
            plt.xlim([0,1])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution on $w(p)$")
        # plt.hist(w_pg_all, bins = 1000)
        sns.kdeplot(w_pg_all)
        plt.xlim([0,1])
        # [plt.axvline(_w, linewidth=1, color='red') for _w in w_pg_means]
        plt.show()

    if plot_weighthyper:
        w_a_ = []
        w_b_ = []
        X    = []

        for l in range(n_trials):
            for j in range(n_samples):
                w_a_.append(weight_a_pg[j,l])
                w_b_.append(weight_b_pg[j,l])

                x = np.linspace(beta.ppf(0.01, weight_a_pg[j,l], weight_b_pg[j,l]), beta.ppf(0.99, weight_a_pg[j,l], weight_b_pg[j,l]), 1000)

                X.append(beta.pdf(x, weight_a_pg[j,l], weight_b_pg[j,l]))
                
        w = []

        for i in range(1000):
            tmp = 0
            for j in range(n_trials*n_samples):
                tmp += X[j][i]
            w.append(tmp/(n_trials*n_samples))
        
        
        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $w(p)$")
        plt.subplot(211)
        plt.title(f"Marginalised beta distribution")
        x = np.linspace(0,1,1000)
        plt.plot(x, w)
        # sns.kdeplot(w)
        plt.subplot(223)
        plt.title(f"Distribution for $\\alpha_w$")
        # plt.hist(w_pg_all, bins = 1000)
        sns.kdeplot(w_a_)
        plt.subplot(224)
        plt.title(f"Distribution for $\\beta_x$")
        # plt.hist(w_pg_all, bins = 1000)
        sns.kdeplot(w_b_)
        plt.show()

    if plot_alphaGain_pg:
        alphaGain_pg_all = []
        alphaGain_pg_means = []
        alphaGain_pg_ = []
        for i in range(n_subjects):
            tmp = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in pg_model:
                        alphaGain_pg_all.append(alphaGain_pg[i,j,l])
                        tmp.append(alphaGain_pg[i,j,l])
            alphaGain_pg_means.append(np.mean(tmp))
            alphaGain_pg_.append(tmp)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for $\\alpha$")
        for j in range(n_subjects):
            plt.subplot(6,3,j+1)
            plt.title(f"Subject {j+1}")
            plt.hist(alphaGain_pg_[j], density = True, bins = 100)
            # plt.axvline(np.mean(alphaGain_pg_[j]),color='red')
            plt.xlim([0,1])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\\alpha$")
        sns.kdeplot(alphaGain_pg_all)
        plt.xlim([0,1])
        # [plt.axvline(_w, linewidth=1, color='red') for _w in alphaGain_pg_means]
        plt.show()

    if plot_alphaGain_pg_hyper:
        alphaGain_pg_mu = []
        alphaGain_pg_tau = []
        alphaGain_pg_sigma = []
        for l in range(n_trials):
            for j in range(n_samples):
                alphaGain_pg_mu.append(np.exp(mu_log_alphaGain_pg[j,l]))
                alphaGain_pg_tau.append(tau_log_alphaGain_pg[j,l])
                alphaGain_pg_sigma.append(sigma_log_alphaGain_pg[j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\\alpha$")
        plt.subplot(211)
        plt.title(f"Distribution for $\mu$")
        sns.kdeplot(alphaGain_pg_mu)
        plt.subplot(223)
        plt.title(f"Distribution for $\\tau$")
        sns.kdeplot(alphaGain_pg_tau)
        plt.xlim(right=4000)
        plt.subplot(224)
        plt.title(f"Distribution for $\sigma$")
        sns.kdeplot(alphaGain_pg_sigma)
        plt.show()

    if plot_beta_pg:
        beta_pg_all_add = []
        beta_pg_means_add = []
        beta_pg_add_ = []
        beta_pg_all_mul = []
        beta_pg_means_mul = []
        beta_pg_mul_ = []
        for i in range(n_subjects):
            tmp_add = []
            tmp_mul = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in pg_model:
                        beta_pg_all_add.append(beta_pg[0,i,j,l])
                        tmp_add.append(beta_pg[0,i,j,l])
                        beta_pg_all_mul.append(beta_pg[1,i,j,l])
                        tmp_mul.append(beta_pg[1,i,j,l])
            beta_pg_means_add.append(np.mean(tmp_add))
            beta_pg_add_.append(tmp_add)
            beta_pg_means_mul.append(np.mean(tmp_mul))
            beta_pg_mul_.append(tmp_mul)

        fig = plt.figure(fig_nr)
        fig_nr += 1
        outer = gridspec.GridSpec(6, 3, wspace=0.2, hspace=1)

        for j in range(n_subjects):
            inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                            subplot_spec=outer[j], wspace=0.2, hspace=0.4)
            
            # set outer titles
            ax = plt.Subplot(fig, outer[j])
            ax.set_title("Subject {}".format(j+1),  y=1.2)
            ax.axis('off')
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[0])
            ax.set_title(f"Additive")
            # ax.set_xlim([-2,50])
            ax.set_yticks([])
            t = ax.hist(beta_pg_add_[j], bins=np.arange(min(beta_pg_add_[j]), max(beta_pg_add_[j]) + 0.5, 0.5), density = True)
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, inner[1])
            t = ax.hist(beta_pg_mul_[j], bins=np.arange(min(beta_pg_mul_[j]), max(beta_pg_mul_[j]) + 0.5, 0.5), density = True)
            ax.set_title(f"Multiplicative")
            ax.set_yticks([])
            # ax.set_xlim([-2,50])
            fig.add_subplot(ax)
        fig.suptitle('Distributions of $\\beta$')

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\\beta$")
        plt.subplot(121)
        plt.title("Additive")
        plt.hist(beta_pg_all_add, bins=np.arange(min(beta_pg_all_add), max(beta_pg_all_add) + 0.5, 0.5), density = True)
        # sns.kdeplot(beta_tw_all_add)
        plt.xlim([-25,400])
        plt.subplot(122)
        plt.title("Multiplikative")
        plt.hist(beta_pg_all_mul, bins=np.arange(min(beta_pg_all_mul), max(beta_pg_all_mul) + 0.5, 0.5), density = True)
        # sns.kdeplot(beta_tw_all_mul)
        plt.xlim([-25,400])
        plt.show()

    if plot_beta_pg_hyper:
        beta_pg_mu_add = []
        beta_pg_mu_mul = []
        beta_pg_tau_add = []
        beta_pg_tau_mul = []
        beta_pg_sigma_add = []
        beta_pg_sigma_mul = []
        for l in range(n_trials):
            for j in range(n_samples):
                beta_pg_mu_add.append(np.exp(mu_log_beta_pg[0,j,l]))
                beta_pg_mu_mul.append(np.exp(mu_log_beta_pg[1,j,l]))
                beta_pg_tau_add.append(sigma_log_beta_pg[0,j,l])
                beta_pg_tau_mul.append(sigma_log_beta_pg[1,j,l])
                beta_pg_sigma_add.append(tau_log_beta_pg[0,j,l])
                beta_pg_sigma_mul.append(tau_log_beta_pg[1,j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\\beta$")
        plt.subplot(221)
        plt.title(f"Distribution for $\mu_a$")
        sns.kdeplot(beta_pg_mu_add)
        plt.subplot(222)
        plt.title(f"Distribution for $\mu_m$")
        sns.kdeplot(beta_pg_mu_mul)
        plt.subplot(245)
        plt.title(f"Distribution for $\\tau_a$")
        sns.kdeplot(beta_pg_tau_add)
        plt.subplot(246)
        plt.title(f"Distribution for $\\tau_m$")
        sns.kdeplot(beta_pg_tau_mul)
        plt.subplot(247)
        plt.title(f"Distribution for $\sigma_a$")
        sns.kdeplot(beta_pg_sigma_add)
        plt.subplot(248)
        plt.title(f"Distribution for $\sigma_m$")
        sns.kdeplot(beta_pg_sigma_mul)
        plt.show()

####### ISO
if plot_iso_params:
    print("\n ISO parameters")
    if plot_iso_eta:
        eta_iso_all = []
        eta_iso_means = []
        eta_iso_ = []
        for i in range(n_subjects):
            tmp = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in iso_model:
                        eta_iso_all.append(np.exp(eta_iso[i,j,l]))
                        tmp.append(np.exp(eta_iso[i,j,l]))
            eta_iso_means.append(np.mean(tmp))
            eta_iso_.append(tmp)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for $\eta$")
        for j in range(n_subjects):
            plt.subplot(6,3,j+1)
            plt.title(f"Subject {j+1}")
            plt.hist(eta_iso_[j], density = True, bins = 100)
            # plt.axvline(np.mean(eta_iso_means[j]),color='red')
            plt.xlim([0,10])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\eta$")
        sns.kdeplot(eta_iso_all)
        # plt.xlim([0,1])
        # [plt.axvline(_w, linewidth=1, color='red') for _w in eta_iso_means]
        plt.show()

    if plot_iso_eta_hypers:
        eta_iso_mu = []
        eta_iso_tau = []
        eta_iso_sigma = []
        for l in range(n_trials):
            for j in range(n_samples):
                eta_iso_mu.append(mu_eta_iso[j,l])
                eta_iso_tau.append(tau_eta_iso[j,l])
                eta_iso_sigma.append(sigma_eta_iso[j,l])

        print(np.shape(eta_iso_mu))

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\eta$")
        plt.subplot(211)
        plt.title(f"Distribution for $\mu$")
        sns.kdeplot(eta_iso_mu)
        plt.subplot(223)
        plt.title(f"Distribution for $\\tau$")
        sns.kdeplot(eta_iso_tau)
        plt.subplot(224)
        plt.title(f"Distribution for $\sigma$")
        sns.kdeplot(eta_iso_sigma)
        plt.show()

    if plot_beta_iso:
        beta_iso_all_add = []
        beta_iso_means_add = []
        beta_iso_add_ = []
        beta_iso_all_mul = []
        beta_iso_means_mul = []
        beta_iso_mul_ = []
        for i in range(n_subjects):
            tmp_add = []
            tmp_mul = []
            for j in range(n_samples):
                for l in range(n_trials):
                    if z[i,j,l] in iso_model:
                        beta_iso_all_add.append(beta_iso[0,i,j,l])
                        tmp_add.append(beta_iso[0,i,j,l])
                        beta_iso_all_mul.append(beta_iso[1,i,j,l])
                        tmp_mul.append(beta_iso[1,i,j,l])
            beta_iso_means_add.append(np.mean(tmp_add))
            beta_iso_add_.append(tmp_add)
            beta_iso_means_mul.append(np.mean(tmp_mul))
            beta_iso_mul_.append(tmp_mul)

        fig = plt.figure(fig_nr)
        fig_nr += 1
        outer = gridspec.GridSpec(6, 3, wspace=0.2, hspace=1)

        for j in range(n_subjects):
            print(j)
            inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                            subplot_spec=outer[j], wspace=0.2, hspace=0.4)
            
            # set outer titles
            ax = plt.Subplot(fig, outer[j])
            ax.set_title("Subject {}".format(j+1),  y=1.2)
            ax.axis('off')
            fig.add_subplot(ax)

            if j not in [0,4,5,9,12,14]:
                ax = plt.Subplot(fig, inner[0])
                ax.set_title(f"Additive")
                # ax.set_xlim([-2,50])
                ax.set_yticks([])
                t = ax.hist(beta_iso_add_[j], bins=np.arange(min(beta_iso_add_[j]), max(beta_iso_add_[j]) + 1, 1), density = True)
                fig.add_subplot(ax)

                ax = plt.Subplot(fig, inner[1])
                t = ax.hist(beta_iso_mul_[j], bins=np.arange(min(beta_iso_mul_[j]), max(beta_iso_mul_[j]) + 1, 1), density = True)
                ax.set_title(f"Multiplicative")
                ax.set_yticks([])
                # ax.set_xlim([-2,50])
                fig.add_subplot(ax)
            else:
                ax = plt.Subplot(fig, inner[0])
                ax.set_title(f"Additive")
                ax.set_yticks([])
                # ax.set_xlim([-2,50])
                t = ax.hist(beta_iso_add_[j], bins=np.arange(min(beta_iso_add_[j]), max(beta_iso_add_[j]) + 0.1, 0.1), density = True)
                fig.add_subplot(ax)

                ax = plt.Subplot(fig, inner[1])
                t = ax.hist(beta_iso_mul_[j], bins=np.arange(min(beta_iso_mul_[j]), max(beta_iso_mul_[j]) + 0.1, 0.1), density = True)
                ax.set_title(f"Multiplicative")
                ax.set_yticks([])
                # ax.set_xlim([-2,50])
                fig.add_subplot(ax)
        fig.suptitle('Distributions of $\\beta$')

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle(f"Marginal distribution for $\\beta$")
        plt.subplot(121) 
        plt.title("Additive")
        print("Additive")
        plt.hist(beta_iso_all_add, bins=np.arange(min(beta_iso_all_add), max(beta_iso_all_add) + 0.5, 0.5), density = True)
        # sns.kdeplot(beta_tw_all_add)
        plt.xlim([-2,50])
        plt.subplot(122)
        plt.title("Multiplikative")
        print("Multiplicative")
        plt.hist(beta_iso_all_mul, bins=np.arange(min(beta_iso_all_mul), max(beta_iso_all_mul) + 0.5, 0.5), density = True)
        # sns.kdeplot(beta_tw_all_mul)
        plt.xlim([-2,50])
        plt.show()

    if plot_beta_iso_hyper:
        beta_iso_mu_add = []
        beta_iso_mu_mul = []
        beta_iso_tau_add = []
        beta_iso_tau_mul = []
        beta_iso_sigma_add = []
        beta_iso_sigma_mul = []
        for l in range(n_trials):
            for j in range(n_samples):
                beta_iso_mu_add.append(np.exp(mu_log_beta_iso[0,j,l]))
                beta_iso_mu_mul.append(np.exp(mu_log_beta_iso[1,j,l]))
                beta_iso_tau_add.append(sigma_log_beta_iso[0,j,l])
                beta_iso_tau_mul.append(sigma_log_beta_iso[1,j,l])
                beta_iso_sigma_add.append(tau_log_beta_iso[0,j,l])
                beta_iso_sigma_mul.append(tau_log_beta_iso[1,j,l])

        plt.figure(fig_nr)
        fig_nr += 1
        plt.suptitle("Distributions for hypers on $\\beta$")
        plt.subplot(221)
        plt.title(f"Distribution for $\mu_a$")
        sns.kdeplot(beta_iso_mu_add)
        plt.subplot(222)
        plt.title(f"Distribution for $\mu_m$")
        sns.kdeplot(beta_iso_mu_mul)
        plt.subplot(245)
        plt.title(f"Distribution for $\\tau_a$")
        sns.kdeplot(beta_iso_tau_add)
        plt.subplot(246)
        plt.title(f"Distribution for $\\tau_m$")
        sns.kdeplot(beta_iso_tau_mul)
        plt.subplot(247)
        plt.title(f"Distribution for $\sigma_a$")
        sns.kdeplot(beta_iso_sigma_add)
        plt.subplot(248)
        plt.title(f"Distribution for $\sigma_m$")
        sns.kdeplot(beta_iso_sigma_mul)
        plt.show()










