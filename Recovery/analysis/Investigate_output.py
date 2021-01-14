import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

file_name = 'JAGS_StrongModels_Subjectwise_ExpandZ_pt_gain_FlatZprior_allData_burn_100_samps_50_chains_4_04-09-2020 21-34'
#file_name = 'JAGS_test_PT_PG_PM_v1_FlatZprior_allData_burn_1000_samps_500_chains_4_04-13-2020 20-49'
#file_name = 'JAGS_test_PT_PG_v3_FlatZprior_allData_burn_1000_samps_500_chains_4_04-03-2020 00-09'
#file_name = 'JAGS_test_PT_PG_PM_v1_FlatZprior_allData_burn_100_samps_50_chains_4_04-15-2020 11-33'
with h5py.File(f'samples_stats/{file_name}.mat', 'r') as file:
    print(list(file.keys()))
    print(list(file['samples'].keys()))
    print(list(file['stats']['Rhat'].keys()))
    alphaGain_pg = file['samples'].get('alphaGain_pg').value
    w_pg = file['samples'].get('w_pg').value
    # w_pm = file['samples'].get('w_pm').value
    z = file['samples'].get('z').value 
    R_z = file['stats']['Rhat'].get('z').value
    R_wpg = file['stats']['Rhat'].get('w_pg').value
    # R_wpm = file['stats']['Rhat'].get('w_pm').value
    # R_apg = file['stats']['Rhat'].get('alphaGain_pg').value


# print(np.shape(alphaGain_pg))
print(np.shape(w_pg))
# print(np.shape(w_pm))
print(np.shape(z))
print(np.shape(R_z))
print(np.shape(R_wpg))
# print(np.shape(R_wpm))
print(R_z)
print(R_wpg)
# print(R_wpm)

# sys.exit(0)

n_subjects = np.shape(w_pg)[0]
n_samples = np.shape(w_pg)[1]
n_trials = np.shape(w_pg)[2]

max_r = max(max(R_z[0]), max(R_wpg[0]), max(R_wpm[0]))
min_r = min(min(R_z[0]), min(R_wpg[0]), min(R_wpm[0]))
max_r_value = 1.1

# ## R-values
# plt.figure(1)
# plt.subplot(311)
# plt.title("R_hat values for z")
# plt.plot(range(n_subjects), R_z[0], 'o')
# plt.xlim(-1,18)
# # plt.ylim([min_r-0.1, max_r+0.1])
# plt.axhline(y=max_r_value, color='r', linestyle='--')
# plt.subplot(312)
# plt.title("R_hat values for w_pg")
# plt.plot(range(n_subjects),R_wpg[0], 'o')
# plt.xlim(-1,18)
# # plt.ylim([min_r-0.1, max_r+0.1])
# plt.axhline(y=max_r_value, color='r', linestyle='--')
# plt.subplot(313)
# plt.title("Rw_pg values for w_pm")
# plt.plot(range(n_subjects),R_wpm[0], 'o')
# plt.xlim(-1,18)
# # plt.ylim([min_r-0.1, max_r+0.1])
# plt.axhline(y=max_r_value, color='r', linestyle='--')
# plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
# plt.show()

# ## W
# for i in range(18):
#     subject_number = i
#     plt.figure(i+2)
#     plt.suptitle(f"w_pg and w_pm for subject {subject_number + 1}")
#     plt.subplot(411)
#     plt.plot(range(n_samples), w_pm[subject_number,:,0], color = "blue")
#     plt.subplot(412)
#     plt.plot(range(n_samples), w_pm[subject_number,:,1], color = "blue")
#     plt.subplot(413)
#     plt.plot(range(n_samples), w_pm[subject_number,:,2], color = "blue")
#     plt.subplot(414)
#     plt.plot(range(n_samples), w_pm[subject_number,:,3], color = "blue")

#     plt.figure(i+2)
#     plt.subplot(411)
#     plt.plot(range(n_samples), w_pg[subject_number,:,0], color = "red")
#     plt.subplot(412)
#     plt.plot(range(n_samples), w_pg[subject_number,:,1], color = "red")
#     plt.subplot(413)
#     plt.plot(range(n_samples), w_pg[subject_number,:,2], color = "red")
#     plt.subplot(414)
#     plt.plot(range(n_samples), w_pg[subject_number,:,3], color = "red")
# plt.show()

# sys.exit(0)

## Z
z_sample_choices_extended = np.empty([n_samples*n_trials , n_subjects])
z_sample_choices = []
for i in range(n_subjects): 
    tmp =  []
    PT = 0
    PG = 0
    PM = 0
    for j in range(n_samples):
        for l in range(n_trials):
            #Append for extended
            tmp.append(z[i,j,l])
            
            #Condence choices
            if z[i,j,l] in [3,6,9,12]:
                PM += 1
            elif z[i,j,l] in [2,5,8,11]:
                PG += 1
            else:
                PT += 1
    
    z_sample_choices_extended[:,i] = tmp
    z_sample_choices.append([PT, PG, PM])


x = ['PT', 'PG', 'PM']
plt.figure(1)
plt.suptitle("Model indicator for each Subject")
for j in range(n_subjects):
    plt.subplot(6,3,j+1)
    plt.title("subject %.0f - Rhat = %.2f" %(j+1, R_z[0][j]))
    plt.bar(x, z_sample_choices[j])
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

plt.figure(2)
plt.suptitle("Model indicator for each Subject")
for j in range(n_subjects):
    plt.subplot(6,3,j+1)
    plt.title(f"subject {j+1}")
    plt.hist(z_sample_choices_extended[:,j])
    plt.xlim([1,12])
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
plt.show()

# sys.exit()

## W-distributions
w_pg_all = []
w_pg_post = []
w_pm_post = []
w_pg_ = []
w_pm_ = []
for i in range(n_subjects):
    tmp_pg_post = []
    tmp_pm_post = []
    tmp_pg = []
    tmp_pm = []
    for j in range(n_samples):
        for l in range(n_trials):
            w_pg_all.append(w_pg[i,j,l])
            tmp_pm_post.append(w_pm[i,j,l])
            if z[i,j,l] in [3,6,9,12]:
                tmp_pm.append(w_pm[i,j,l])
            elif z[i,j,l] in [2,5,8,11]:
                tmp_pg.append(w_pg[i,j,l])
    w_pg_post.append(tmp_pg_post)
    w_pm_post.append(tmp_pm_post)
    w_pg_.append(tmp_pg)
    w_pm_.append(tmp_pm)

plt.figure(3)
plt.suptitle("Subjective weights PG posterior distribution")
for j in range(n_subjects):
    plt.subplot(6,3,j+1)
    plt.title(f"Subject %.0f - R value = %.2f" %(j+1, R_wpg[0][j]))
    plt.hist(w_pg_post[j])
    plt.axvline(np.mean(w_pg_post[j]),color='red')
    plt.xlim([0,1])
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

plt.figure(4)
plt.suptitle("Subjective weights PM posterior distribution")
for j in range(n_subjects):
    plt.subplot(6,3,j+1)
    plt.title(f"Subject %.0f - R value = %.2f" %(j+1, R_wpm[0][j]))
    plt.hist(w_pm_post[j])
    plt.axvline(np.mean(w_pm_post[j]),color='red')
    plt.xlim([0,1])
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

plt.figure(5)
plt.suptitle("Subjective weights for Weighted gain")
for j in range(n_subjects):
    plt.subplot(6,3,j+1)
    plt.title(f"Subject {j+1} - Number chosen: {len(w_pg_[j])}")
    plt.hist(w_pg_[j])
    plt.axvline(np.mean(w_pg_[j]),color='red')
    plt.xlim([0,1])
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

plt.figure(6)
plt.suptitle("Subjective weights for Weighted mixed")
for j in range(n_subjects):
    plt.subplot(6,3,j+1)
    plt.title(f"Subject {j+1} - Number chosen: {len(w_pm_[j])}")
    plt.hist(w_pm_[j])
    plt.axvline(np.mean(w_pm_[j]),color='red')
    plt.xlim([0,1])
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
plt.show()

plt.figure(7)
plt.hist(w_pg_all, bins = 100)
plt.show()
