import torch
import numpy as np

results_baseline = [
    ["pwcWOX1_cc", 1.9744 , 11.4516, 5.7697 , 17.1193, 1, 0, 0, 0, 1, 0, 0, 0],
    ["pwcWOX1_tt", 3.1599 , 7.9826 , 4.9328 , 16.3024, 0, 1, 0, 0, 0, 1, 0, 0],
    ["pwcWOX1_ss", 3.728  , 13.7999, 5.8259 , 13.9545, 0, 0, 1, 0, 0, 0, 1, 0],
    ["pwcWOX1_kk" , 10.2856, 32.0227, 13.6827, 21.6806, 0, 0, 0, 1, 0, 0, 0, 1]
]

results_header = ["model_id", "evalC", "evalT", "evalS", "evalK", "encC", "encT", "encS", "encK", "decC", "decT", "decS", "decK"]
results = {
    "pwcWOX1_cc": ["pwcWOX1_cc", 1.9744 , 11.4516, 5.7697 , 17.1193, 1, 0, 0, 0, 1, 0, 0, 0],
    "pwcWOX1_tt": ["pwcWOX1_tt", 3.1599 , 7.9826 , 4.9328 , 16.3024, 0, 1, 0, 0, 0, 1, 0, 0],
    "pwcWOX1_ss": ["pwcWOX1_ss", 3.728  , 13.7999, 5.8259 , 13.9545, 0, 0, 1, 0, 0, 0, 1, 0],
    "pwcWOX1_kk": ["pwcWOX1_kk" , 10.2856, 32.0227, 13.6827, 21.6806, 0, 0, 0, 1, 0, 0, 0, 1],
    "WOX1Blind_ct": ["WOX1Blind_ct"  , 3.1117 , 9.739  , 5.5985 , 12.5218, 1, 0, 0, 0, 0, 1, 0, 0],
    "WOX1Blind_cs": ["WOX1Blind_cs"  , 3.1059 , 13.8979, 5.8513 , 14.5128, 1, 0, 0, 0, 0, 0, 1, 0],
    "WOX1Blind_ck": ["WOX1Blind_ck"  , 8.1603 , 28.5486, 11.8594, 22.9861, 1, 0, 0, 0, 0, 0, 0, 1],
    "WOX1Blind_tc": ["WOX1Blind_tc"  , 3.2556 , 11.3651, 6.6496 , 23.3206, 0, 1, 0, 0, 1, 0, 0, 0],
    "WOX1Blind_ts": ["WOX1Blind_ts"  , 4.8672 , 13.0559, 7.3878 , 21.828 , 0, 1, 0, 0, 0, 0, 1, 0],
    "WOX1Blind_tk": ["WOX1Blind_tk"  , 9.6237 , 26.7064, 13.4904, 25.1068, 0, 1, 0, 0, 0, 0, 0, 1],
    "WOX1Blind_sc": ["WOX1Blind_sc"  , 2.8337 , 13.1091, 6.3045 , 18.2382, 0, 0, 1, 0, 1, 0, 0, 0],
    "WOX1Blind_st": ["WOX1Blind_st"  , 4.0227 , 12.8012, 6.4953 , 13.621 , 0, 0, 1, 0, 0, 1, 0, 0],
    "WOX1Blind_sk": ["WOX1Blind_sk"  , 8.1455 , 28.2851, 11.3799, 23.6997, 0, 0, 1, 0, 0, 0, 0, 1],
    "WOX1Blind_kc": ["WOX1Blind_kc"  , 16.512 , 34.078 , 15.1191, 25.4539, 0, 0, 0, 1, 1, 0, 0, 0],
    "WOX1Blind_kt": ["WOX1Blind_kt"  , 25.901 , 57.7069, 31.1695, 21.8509, 0, 0, 0, 1, 0, 1, 0, 0],
    "WOX1Blind_ks": ["WOX1Blind_ks"  , 9.0989 , 25.5259, 11.4738, 25.2839, 0, 0, 0, 1, 0, 0, 1, 0]
}


def filter_for_dataset(dataset_char):
    return [row for key, row in results.items() if (key[-2] == dataset_char) or (key[-1] == dataset_char)]


char_map = {
    "c": 0,
    "t": 1,
    "s": 2,
    "k": 3,
}

num_vars = 8
num_consts = 1

def build_row(row, dataset_id, no_b=False):
    a = np.zeros(4 * num_vars + num_consts, dtype=np.double)
    a_idx = char_map[row[0][-2]]
    b_idx = char_map[row[0][-1]]
    aa = results_baseline[a_idx][1 + dataset_id]
    bb = results_baseline[b_idx][1 + dataset_id]
    a[0 + dataset_id] = aa
    a[4 + dataset_id] = aa * aa
    a[8 + dataset_id] = bb
    a[12 + dataset_id] = bb * bb
    a[16 + dataset_id] = aa * bb
    a[20 + dataset_id] = aa * aa * bb
    a[24 + dataset_id] = aa * bb * bb
    a[28 + dataset_id] = aa * aa * bb * bb
    #a[32 + dataset_id] = np.exp(aa)
    #a[36 + dataset_id] = np.exp(bb)
    #a[-4] = 1  # constant term  per dataset ...
    #a[-3] = 1  # constant term  per dataset ...
    #a[-2] = 1  # constant term  per dataset ...
    a[-1] = 1  # constant term  per dataset ...

    b = row[1 + dataset_id]
    if no_b:
        return a
    else:
        return a, b


def build_combined_row(row, dataset_id, no_b=False):
    a = np.zeros(num_vars + 1, dtype=np.double)
    a_idx = char_map[row[0][-2]]
    b_idx = char_map[row[0][-1]]
    aa = results_baseline[a_idx][1 + dataset_id]
    bb = results_baseline[b_idx][1 + dataset_id]
    a[0] = aa
    a[1] = aa * aa
    a[2] = bb
    a[3] = bb * bb
    a[4] = aa * bb
    a[5] = aa * aa * bb
    a[6] = aa * bb * bb
    a[7] = aa * aa * bb * bb
    #a[32 + dataset_id] = np.exp(aa)
    #a[36 + dataset_id] = np.exp(bb)
    a[-1] = 1  # constant term ...

    b = row[1 + dataset_id]
    if no_b:
        return a
    else:
        return a, b


row_builder_func = build_row
row_factor = 4
#row_builder_func = build_combined_row
#row_factor = 1

results_set = filter_for_dataset("c")
a = np.zeros((len(results_set)*4, row_factor*num_vars+num_consts), dtype=np.double)
b = np.zeros((len(results_set)*4), dtype=np.double)
for i, row in enumerate(results_set):

    for dataset_id in range(4):
        ra, rb = row_builder_func(row, dataset_id)
        a[i * 4 + dataset_id, :] = ra
        b[i * 4 + dataset_id] = rb


solution, residuals, rank, s = np.linalg.lstsq(a,b)
print(">>", solution)


print("=======")
row = results_set[0]
print(np.dot(row_builder_func(row, 0, True), solution), row[1])
print(np.dot(row_builder_func(row, 1, True), solution), row[2])
print(np.dot(row_builder_func(row, 2, True), solution), row[3])
print(np.dot(row_builder_func(row, 3, True), solution), row[4])
