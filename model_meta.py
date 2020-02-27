
model_meta_fields = ["model", "experiment", "base_model", "dataset_base", "dataset_fine", "folder_name", "loader_"]
model_meta = {
    # baseline_single pwc
    "A": ["PWCNet", "baseline_single", None, "chairs", None, "A_PWCNet-onChairs-20191121-171532", None],
    "I": ["PWCNet", "baseline_single", None, "things", None, "I_PWCNet-things_20191209-131019", None],
    "H": ["PWCNet", "baseline_single", None, "sintel", None, "H_PWCNet-sintel-20191209-150448", None],
    "W": ["PWCNet", "baseline_single", None, "kitti", None, "W_PWCNet-kitti-20191216-124247", None],
    "pwc_kitti_temp": ["PWCNet", "baseline_single", None, "kitti", None, "PWCNet-kitti_fixed_aug-20200225-020620", None],
    # baseline_fine pwc
    "F": ["PWCNet", "baseline_fine", "A", "chairs", "chairs", "F_PWCNet-A_fine_chairs-20191212-133136", None],
    "K": ["PWCNet", "baseline_fine", "A", "chairs", "things", "K_PWCNet-A_fine_things-20191212-133436", None],
    "R": ["PWCNet", "baseline_fine", "A", "chairs", "sintel", "R_PWCNet-A_fine_sintel-20191218-135407", None],
    "S": ["PWCNet", "baseline_fine", "A", "chairs", "kitti", "S_PWCNet-A_fine_KITTI-20191216-125450", None],
    "V": ["PWCNet", "baseline_fine", "I", "things", "chairs", "V_PWCNet-I_fine_chairs-20191230-031321", None],
    "Y": ["PWCNet", "baseline_fine", "I", "things", "things", "Y_PWCNet-I_fine_things-20191230-024005", None],
    "X": ["PWCNet", "baseline_fine", "I", "things", "sintel", "X_PWCNet-I_fine_sintel-20191227-155229", None],
    "O": ["PWCNet", "baseline_fine", "I", "things", "kitti", "O_PWCNet-I_fine_kitti-20191226-230605", None],

    # baseline_single flowet
    "D": ["FlowNet1S", "baseline_single", None, "chairs", None, "D_FlowNet1S-onChairs-20191205-145310", None],
    "E": ["FlowNet1S", "baseline_single", None, "things", None, "E_FlowNet1S-onThings-20191205-115159", None],
    "M": ["FlowNet1S", "baseline_single", None, "sintel", None, "M_FlowNet1S-sintel-20191220-111613", None],
    "U": ["FlowNet1S", "baseline_single", None, "kitti", None, "U_FlowNet1S-kitti-20191220-110932", None],
    # baseline_fine flownet
    "L": ["FlowNet1S", "baseline_fine", "D", "chairs", "chairs", "L_FlowNet1S-D_fine_chairs-20191218-134755", None],
    "J": ["FlowNet1S", "baseline_fine", "D", "chairs", "things", "J_FlowNet1S-D_fine_things-20191216-130336", None],
    "N": ["FlowNet1S", "baseline_fine", "D", "chairs", "sintel", "N_FlowNet1S-D_fine_sintel-20191218-140407", None],
    "P": ["FlowNet1S", "baseline_fine", "D", "chairs", "kitti", "P_FlowNet1S-D_fine_kitti-20191227-005059", None],
    "ZZ": ["FlowNet1S", "baseline_fine", "E", "things", "chairs", "ZZ_FlowNet1S-E_fine_chairs-20200111-180519", None],
    "ZT": ["FlowNet1S", "baseline_fine", "E", "things", "things", "ZT_FlowNet1S-E_fine_things_res1-20200116-140705", None],
    "Z": ["FlowNet1S", "baseline_fine", "E", "things", "sintel", "Z_FlowNet1S-sintel-20191227-163340", None],
    "Q": ["FlowNet1S", "baseline_fine", "E", "things", "kitti", "Q_FlowNet1S-E_fine_kitti-20191227-020409", None],

    # baseline_single pwc repeated
    "C": ["PWCNet", "baseline_single_repeated", None, "chairs", None, "C_PWCNet-onChairs-20191126-113818", None],
    "I2": ["PWCNet", "baseline_single_repeated", None, "things", None, "I2_PWCNet-things-20191230-022450", None],
    "H2": ["PWCNet", "baseline_single_repeated", None, "sintel", None, "H2_PWCNet-sintel-20191227-162133", None],
    # baseline_single flownet repeated
    "T": ["FlowNet1S", "baseline_single_repeated", None, "chairs", None, "T_FlowNet1S-sintel-20191227-163340", None],

    # baseline_fine pwc repeated
    "G": ["PWCNet", "baseline_fine_repeated", "A", "chairs", "sintel", "G_PWCNet-A_fine_sintel-20191212-134449", None],

    # baseline_fine pwc special
    "NF01": ["PWCNet", "baseline_fine_special", "H", "sintel", "sintel", "NF01_PWCNet-H_fine_sintel-20191230-021221", None],


    # fusing blind
    "blind_ah": ["PWCNet", "fused_blind", "A,H", "chairs,sintel", None, "blind@ah", None],
    "blind_ai": ["PWCNet", "fused_blind", "A,I", "chairs,things", None, "blind@ai", None],
    "blind_aw": ["PWCNet", "fused_blind", "A,W", "chairs,kitti", None, "blind@aw", None],
    "blind_ha": ["PWCNet", "fused_blind", "H,A", "sintel,chairs", None, "blind@ha", None],
    "blind_hi": ["PWCNet", "fused_blind", "H,I", "sintel,things", None, "blind@hi", None],
    "blind_hw": ["PWCNet", "fused_blind", "H,W", "sintel,kitti", None, "blind@hw", None],
    "blind_ia": ["PWCNet", "fused_blind", "I,A", "things,chairs", None, "blind@ia", None],
    "blind_ih": ["PWCNet", "fused_blind", "I,H", "things,sintel", None, "blind@ih", None],
    "blind_iw": ["PWCNet", "fused_blind", "I,W", "things,kitti", None, "blind@iw", None],
    "blind_wa": ["PWCNet", "fused_blind", "W,A", "kitti,chairs", None, "blind@wa", None],
    "blind_wh": ["PWCNet", "fused_blind", "W,H", "kitti,sintel", None, "blind@wh", None],
    "blind_wi": ["PWCNet", "fused_blind", "W,I", "kitti,things", None, "blind@wi", None],
    # fusing blind finetuned
    "000": ["PWCNet", "fused_blind_fine", "A,I", "chairs,things", "sintel", "000_PWCNet-AI_fine_sintel-20191220-133049", None],
    "001": ["PWCNet", "fused_blind_fine", "I,A", "things,chairs", "sintel", "001_PWCNet-IA_fine_sintel-20191220-134913", None],
    "002": ["PWCNet", "fused_blind_fine", "A,H", "chairs,sintel", "sintel", "002_PWCNet-AH_fine_sintel-20191225-203650", None],
    "003": ["PWCNet", "fused_blind_fine", "H,A", "sintel,chairs", "sintel", "003_PWCNet-HA_fine_sintel-20191225-203955", None],
    "004": ["PWCNet", "fused_blind_fine", "I,H", "things,sintel", "sintel", "004_PWCNet-IH_fine_sintel-20191225-204252", None],
    "005": ["PWCNet", "fused_blind_fine", "H,I", "sintel,things", "sintel", "005_PWCNet-HI_fine_sintel-20191225-204931", None],
    # fusing conv33 not trained
    "conv33_ah": ["PWCNetConv33Fusion", "fused_conv33", "A,H", "chairs,sintel", None, "convBlind@ah", None],
    "conv33_ai": ["PWCNetConv33Fusion", "fused_conv33", "A,I", "chairs,things", None, "convBlind@ai", None],
    "conv33_aw": ["PWCNetConv33Fusion", "fused_conv33", "A,W", "chairs,kitti", None, "convBlind@aw", None],
    "conv33_ha": ["PWCNetConv33Fusion", "fused_conv33", "H,A", "sintel,chairs", None, "convBlind@ha", None],
    "conv33_hi": ["PWCNetConv33Fusion", "fused_conv33", "H,I", "sintel,things", None, "convBlind@hi", None],
    "conv33_hw": ["PWCNetConv33Fusion", "fused_conv33", "H,W", "sintel,kitti", None, "convBlind@hw", None],
    "conv33_ia": ["PWCNetConv33Fusion", "fused_conv33", "I,A", "things,chairs", None, "convBlind@ia", None],
    "conv33_ih": ["PWCNetConv33Fusion", "fused_conv33", "I,H", "things,sintel", None, "convBlind@ih", None],
    "conv33_iw": ["PWCNetConv33Fusion", "fused_conv33", "I,W", "things,kitti", None, "convBlind@iw", None],
    "conv33_wa": ["PWCNetConv33Fusion", "fused_conv33", "W,A", "kitti,chairs", None, "convBlind@wa", None],
    "conv33_wh": ["PWCNetConv33Fusion", "fused_conv33", "W,H", "kitti,sintel", None, "convBlind@wh", None],
    "conv33_wi": ["PWCNetConv33Fusion", "fused_conv33", "W,I", "kitti,things", None, "convBlind@wi", None],
    # fusing conv33 finetuned
    "0C300": ["PWCNetConv33Fusion", "fused_conv33_fine", "I,A", "things,chairs", "chairs", "0C300_PWCNetConv33Fusion-fine_chairs-20200105-171055", None],
    "0C301": ["PWCNetConv33Fusion", "fused_conv33_fine", "A,I", "chairs,things", "chairs", "0C301_PWCNetConv33Fusion-fine_chairs-20200105-171258", None],
    "0C302": ["PWCNetConv33Fusion", "fused_conv33_fine", "H,A", "sintel,chairs", "chairs", "0C302_PWCNetConv33Fusion-fine_chairs-20200105-171550", None],
    "0C303": ["PWCNetConv33Fusion", "fused_conv33_fine", "A,H", "chairs,sintel", "chairs", "0C303_PWCNetConv33Fusion-fine_chairs-20200105-171857", None],
    "0C304": ["PWCNetConv33Fusion", "fused_conv33_fine", "H,I", "sintel,things", "chairs", "0C304_PWCNetConv33Fusion-fine_chairs-20200111-174643", None],
    "0C305": ["PWCNetConv33Fusion", "fused_conv33_fine", "I,H", "things,sintel", "chairs", "0C305_res2_PWCNetConv33Fusion-fine_chairs-20200116-141744", None],
    "0C306": ["PWCNetConv33Fusion", "fused_conv33_fine", "W,A", "kitti,chairs", "chairs", "0C306_PWCNetConv33Fusion-fine_chairs-20200105-172316", None],
    "0C307": ["PWCNetConv33Fusion", "fused_conv33_fine", "A,W", "chairs,kitti", "chairs", "0C307_PWCNetConv33Fusion-fine_chairs-20200111-174308", None],
    "0C308": ["PWCNetConv33Fusion", "fused_conv33_fine", "W,I", "kitti,things", "chairs", "0C308_PWCNetConv33Fusion-fine_chairs-20200116-020142", None],
    "0C309": ["PWCNetConv33Fusion", "fused_conv33_fine", "I,W", "things,kitti", "chairs", "0C309_PWCNetConv33Fusion-fine_chairs-20200116-141109", None],
    "0C310": ["PWCNetConv33Fusion", "fused_conv33_fine", "W,H", "kitti,sintel", "chairs", "0C310_PWCNetConv33Fusion-fine_chairs-20200116-022159", None],
    "0C311": ["PWCNetConv33Fusion", "fused_conv33_fine", "H,W", "sintel,kitti", "chairs", "0C311_PWCNetConv33Fusion-fine_chairs-20200122-163116", None],

    # baseline models PWC no fine - x1 zero
    "x1ZeroBlind_A": ["PWCNetX1Zero", "x1_zero_baseline", "A", "chairs", None, "A_PWCNet-onChairs-20191121-171532", None],
    "x1ZeroBlind_I": ["PWCNetX1Zero", "x1_zero_baseline", "I", "things", None, "I_PWCNet-things_20191209-131019", None],
    "x1ZeroBlind_H": ["PWCNetX1Zero", "x1_zero_baseline", "H", "sintel", None, "H_PWCNet-sintel-20191209-150448", None],
    "x1ZeroBlind_W": ["PWCNetX1Zero", "x1_zero_baseline", "W", "kitti", None, "W_PWCNet-kitti-20191216-124247", None],
    # baseline models PWC finetuned - x1 zero
    "x1ZeroBlind_F": ["PWCNetX1Zero", "x1_zero_baseline_fine", "F", "chairs", "chairs", "F_PWCNet-A_fine_chairs-20191212-133136", None],
    "x1ZeroBlind_K": ["PWCNetX1Zero", "x1_zero_baseline_fine", "K", "chairs", "things", "K_PWCNet-A_fine_things-20191212-133436", None],
    "x1ZeroBlind_R": ["PWCNetX1Zero", "x1_zero_baseline_fine", "R", "chairs", "sintel", "R_PWCNet-A_fine_sintel-20191218-135407", None],
    "x1ZeroBlind_S": ["PWCNetX1Zero", "x1_zero_baseline_fine", "S", "chairs", "kitti", "S_PWCNet-A_fine_KITTI-20191216-125450", None],
    "x1ZeroBlind_V": ["PWCNetX1Zero", "x1_zero_baseline_fine", "V", "things", "chairs", "V_PWCNet-I_fine_chairs-20191230-031321", None],
    "x1ZeroBlind_Y": ["PWCNetX1Zero", "x1_zero_baseline_fine", "Y", "things", "things", "Y_PWCNet-I_fine_things-20191230-024005", None],
    "x1ZeroBlind_X": ["PWCNetX1Zero", "x1_zero_baseline_fine", "X", "things", "sintel", "X_PWCNet-I_fine_sintel-20191227-155229", None],
    "x1ZeroBlind_O": ["PWCNetX1Zero", "x1_zero_baseline_fine", "O", "things", "kitti", "O_PWCNet-I_fine_kitti-20191226-230605", None],

    # fused blind x1 zero
    "x1ZeroBlind_ah": ["PWCNetX1Zero", "x1_zero_baseline_fused", "A,H", "chairs,sintel", None, "blind@ah", None],
    "x1ZeroBlind_ai": ["PWCNetX1Zero", "x1_zero_baseline_fused", "A,I", "chairs,things", None, "blind@ai", None],
    "x1ZeroBlind_aw": ["PWCNetX1Zero", "x1_zero_baseline_fused", "A,W", "chairs,kitti", None, "blind@aw", None],
    "x1ZeroBlind_ha": ["PWCNetX1Zero", "x1_zero_baseline_fused", "H,A", "sintel,chairs", None, "blind@ha", None],
    "x1ZeroBlind_hi": ["PWCNetX1Zero", "x1_zero_baseline_fused", "H,I", "sintel,things", None, "blind@hi", None],
    "x1ZeroBlind_hw": ["PWCNetX1Zero", "x1_zero_baseline_fused", "H,W", "sintel,kitti", None, "blind@hw", None],
    "x1ZeroBlind_ia": ["PWCNetX1Zero", "x1_zero_baseline_fused", "I,A", "things,chairs", None, "blind@ia", None],
    "x1ZeroBlind_ih": ["PWCNetX1Zero", "x1_zero_baseline_fused", "I,H", "things,sintel", None, "blind@ih", None],
    "x1ZeroBlind_iw": ["PWCNetX1Zero", "x1_zero_baseline_fused", "I,W", "things,kitti", None, "blind@iw", None],
    "x1ZeroBlind_wa": ["PWCNetX1Zero", "x1_zero_baseline_fused", "W,A", "kitti,chairs", None, "blind@wa", None],
    "x1ZeroBlind_wh": ["PWCNetX1Zero", "x1_zero_baseline_fused", "W,H", "kitti,sintel", None, "blind@wh", None],
    "x1ZeroBlind_wi": ["PWCNetX1Zero", "x1_zero_baseline_fused", "W,I", "kitti,things", None, "blind@wi", None],

    # baseline pwc without X1 connection
    "pwcWOX1_chairs": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "chairs", None, "WOX1_chairs_PWCNetWOX1Connection-20200122-164023", None],
    "pwcWOX1_things": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "things", None, "WOX1_PWCNetWOX1Connection-things-20200127-234143", None],
    "pwcWOX1_sintel": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "sintel", None, "WOX1_PWCNetWOX1Connection-sintel-20200127-232828", None],
    "pwcWOX1_kitti": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "WOX1_PWCNetWOX1Connection-kitti-20200128-000101", None],
    "pwcWOX1_kitti_temp": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_fixed_aug-20200225-090042", None],
    "pwcWOX1_kitti_tempA": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_fixed_aug-20200225-090042/checkpoint_iter_00450.ckpt", None],
    "pwcWOX1_kitti_tempB": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_fixed_aug-20200225-090042/checkpoint_iter_01050.ckpt", None],
    "pwcWOX1_kitti_tempC": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_fixed_aug-20200225-090042/checkpoint_iter_01950.ckpt", None],
    "pwcWOX1_kitti_tempD": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_fixed_aug-20200225-090042/checkpoint_iter_03000.ckpt", None],
    "pwcWOX1_kitti_tempE": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_fixed_aug-20200225-090042/checkpoint_latest.ckpt", None],

    # PWCNetWOX1 finetuned
    "pwcWOX1_chairs_fine_things": ["PWCNetWOX1Connection", "without_x1_connection_finetuned", "pwcWOX1_chairs", "chairs", "things", "PWCNetWOX1Connection-WOX1Chairs_fine_things-20200206-153903", None],

    # pwc without x1 connection between encoder, decoder and context network
    "WOX1Ext_chairs": ["PWCNetWOX1ConnectionExt", "without_x1_connection_ext_baseline", None, "chairs", None, "WOX1Ext_PWCNetWOX1ConnectionExt-onChairs-20200209-181206", None],

    # blind fused pwc without X1 connection
    "WOX1Blind_ct": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_chairs,pwcWOX1_things", "chairs,things", None, "WOX1Blind@ct", None],
    "WOX1Blind_cs": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_chairs,pwcWOX1_sintel", "chairs,sintel", None, "WOX1Blind@cs", None],
    "WOX1Blind_ck": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_chairs,pwcWOX1_kitti", "chairs,kitti", None, "WOX1Blind@ck", None],
    "WOX1Blind_tc": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_things,pwcWOX1_chairs", "things,chairs", None, "WOX1Blind@tc", None],
    "WOX1Blind_ts": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_things,pwcWOX1_sintel", "things,sintel", None, "WOX1Blind@ts", None],
    "WOX1Blind_tk": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_things,pwcWOX1_kitti", "things,kitti", None, "WOX1Blind@tk", None],
    "WOX1Blind_sc": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_sintel,pwcWOX1_chairs", "sintel,chairs", None, "WOX1Blind@sc", None],
    "WOX1Blind_st": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_sintel,pwcWOX1_things", "sintel,things", None, "WOX1Blind@st", None],
    "WOX1Blind_sk": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_sintel,pwcWOX1_kitti", "sintel,kitti", None, "WOX1Blind@sk", None],
    "WOX1Blind_kc": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_kitti,pwcWOX1_chairs", "kitti,chairs", None, "WOX1Blind@kc", None],
    "WOX1Blind_kt": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_kitti,pwcWOX1_things", "kitti,things", None, "WOX1Blind@kt", None],
    "WOX1Blind_ks": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_kitti,pwcWOX1_sintel", "kitti,sintel", None, "WOX1Blind@ks", None],

    # pwc expert model split
    "expert_split02_known": ["CTSKPWCExpertNet02", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet-20200124-000701", "error"],
    "expert_split02_expert0": ["CTSKPWCExpertNet02", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet-20200124-000701", "expert0"],
    "expert_split02_expert1": ["CTSKPWCExpertNet02", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet-20200124-000701", "expert1"],
    "expert_split02_expert2": ["CTSKPWCExpertNet02", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet-20200124-000701", "expert2"],
    "expert_split02_expert3": ["CTSKPWCExpertNet02", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet-20200124-000701", "expert3"],

    # pwc expert model add
    "expert_add01_known":   ["CTSKPWCExpertNetAdd01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "error"],
    "expert_add01_no_expert": ["CTSKPWCExpertNetAdd01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "noExpert"],
    "expert_add01_expert0": ["CTSKPWCExpertNetAdd01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert0"],
    "expert_add01_expert1": ["CTSKPWCExpertNetAdd01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert1"],
    "expert_add01_expert2": ["CTSKPWCExpertNetAdd01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert2"],
    "expert_add01_expert3": ["CTSKPWCExpertNetAdd01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert3"],

    # pwc expert model CTS add
    "expert_CTS_add01_no_expert": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "noExpert"],
    "expert_CTS_add01_expert0": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert0"],
    "expert_CTS_add01_expert1": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert1"],
    "expert_CTS_add01_expert2": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert2"],

    # pwc trained on multiple datasets
    "pwc_on_CTSK": ["PWCNet", "pwc_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_noExpert_PWCNet-20200127-234847", None],

    # pwc iteration inspection
    "pwc_chairs_iter_148": ["PWCNet", "pwc_iteration_inspection", None, "chairs", None, "iter_PWCNet-chairs_148/checkpoint_iter_148.ckpt", None],

    # pwc down sampling encoder
    "pwcDSEncoder_chairs": ["PWCNetDSEncoder", "pwc_ds_encoder", None, "chairs", None, "PWC_DSEncoder_PWCNetDSEncoder-onChairs-20200209-190403", None]

}

model_meta_ordering = [
    # baseline_single pwc
    "A", "I", "H", "W", "pwc_kitti_temp",
    # baseline_fine pwc
    "F", "K", "R", "S", "V", "Y", "X", "O",
    # baseline_single flowet
    "D", "E", "M", "U",
    # baseline_fine flownet
    "L", "J", "N", "P", "ZZ", "ZT", "Z", "Q",
    # baseline_single pwc repeated
    "C", "I2", "H2", 
    # baseline_single flownet repeated
    "T", 
    # baseline_fine pwc repeated
    "G", 
    # baseline_fine pwc special
    "NF01", 
    # fusing blind
    "blind_ah", "blind_ai", "blind_aw", "blind_ha", "blind_hi", "blind_hw", "blind_ia", "blind_ih", "blind_iw",
    "blind_wa", "blind_wh", "blind_wi",
    # fusing blind finetuned
    "000", "001", "002", "003", "004", "005", 
    # fusing conv33 not trained
    "conv33_ah", "conv33_ai", "conv33_aw", "conv33_ha", "conv33_hi", "conv33_hw", "conv33_ia", "conv33_ih",
    "conv33_iw", "conv33_wa", "conv33_wh", "conv33_wi",
    # fusing conv33 finetuned
    "0C300", "0C301", "0C302", "0C303", "0C304", "0C305", "0C306", "0C307", "0C308", "0C309", "0C310", "0C311",
    # baseline models PWC no fine - x1 zero
    "x1ZeroBlind_A", "x1ZeroBlind_I", "x1ZeroBlind_H", "x1ZeroBlind_W",
    # baseline models PWC finetuned - x1 zero
    "x1ZeroBlind_F", "x1ZeroBlind_K", "x1ZeroBlind_R", "x1ZeroBlind_S", "x1ZeroBlind_V", "x1ZeroBlind_Y",
    "x1ZeroBlind_X", "x1ZeroBlind_O",
    # fused blind x1 zero
    "x1ZeroBlind_ah", "x1ZeroBlind_ai", "x1ZeroBlind_aw", "x1ZeroBlind_ha", "x1ZeroBlind_hi", "x1ZeroBlind_hw",
    "x1ZeroBlind_ia", "x1ZeroBlind_ih", "x1ZeroBlind_iw", "x1ZeroBlind_wa", "x1ZeroBlind_wh", "x1ZeroBlind_wi",
    # baseline pwc without X1 connection
    "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti", "pwcWOX1_kitti_temp",
    "pwcWOX1_kitti_tempA", "pwcWOX1_kitti_tempB", "pwcWOX1_kitti_tempC", "pwcWOX1_kitti_tempD", "pwcWOX1_kitti_tempE",
    # PWCNetWOX1 finetuned
    "pwcWOX1_chairs_fine_things",
    # pwc without x1 connection between encoder, decoder and context network
    "WOX1Ext_chairs",
    # blind fused pwc without X1 connection
    "WOX1Blind_ct", "WOX1Blind_cs", "WOX1Blind_ck", "WOX1Blind_tc", "WOX1Blind_ts", "WOX1Blind_tk",
    "WOX1Blind_sc", "WOX1Blind_st", "WOX1Blind_sk", "WOX1Blind_kc", "WOX1Blind_kt", "WOX1Blind_ks",
    # pwc expert model split
    "expert_split02_known",
    "expert_split02_expert0", "expert_split02_expert1", "expert_split02_expert2", "expert_split02_expert3",
    # pwc expert model add
    "expert_add01_known", "expert_add01_no_expert",
    "expert_add01_expert0", "expert_add01_expert1", "expert_add01_expert2", "expert_add01_expert3",
    # pwc expert model CTS add
    "expert_CTS_add01_no_expert",
    "expert_CTS_add01_expert0", "expert_CTS_add01_expert1", "expert_CTS_add01_expert2",
    # pwc trained on multiple datasets
    "pwc_on_CTSK",
    # pwc iteration inspection
    "pwc_chairs_iter_148",
    # pwc down sampling encoder
    "pwcDSEncoder_chairs"
]

model_folders = {
    "_default": "/data/dataB/models/",
    "blind": "/data/dataB/fusedModels_blind/",
    "convBlind": "/data/dataB/fusedModelsConv33/",
    "WOX1Blind": "/data/dataB/fusedModelsWOX1Conn_blind/"
}

# print(set(iter(model_meta.keys())).difference(model_meta_ordering))
assert(len(model_meta) == len(model_meta_ordering))
