
model_meta_fields = ["model", "experiment", "base_model", "dataset_base", "dataset_fine", "folder_name", "loader_", "is_base_model"]
model_meta = {
    # baseline_single pwc
    "pwc_chairs": ["PWCNet", "baseline_single", None, "chairs", None, "A_PWCNet-onChairs-20191121-171532", None, True],
    "I": ["PWCNet", "baseline_single", None, "things", None, "I_PWCNet-things_20191209-131019", None, True],
    "H": ["PWCNet", "baseline_single", None, "sintel", None, "H_PWCNet-sintel-20191209-150448", None, True],
    "pwc_kitti": ["PWCNet", "baseline_single", None, "kitti", None, "pwc_kitti_PWCNet-kitti_fixed_aug-20200225-020620", None, True],
    #"W": ["PWCNet", "baseline_single", None, "kitti", None, "W_PWCNet-kitti-20191216-124247", None, False],
    #"pwc_kitti_temp": ["PWCNet", "baseline_single", None, "kitti", None, "PWCNet-kitti_fixed_aug-20200225-020620", None, False],
    # baseline_fine pwc
    "F": ["PWCNet", "baseline_fine", "pwc_chairs", "chairs", "chairs", "F_PWCNet-A_fine_chairs-20191212-133136", None, False],
    "K": ["PWCNet", "baseline_fine", "pwc_chairs", "chairs", "things", "K_PWCNet-A_fine_things-20191212-133436", None, False],
    "R": ["PWCNet", "baseline_fine", "pwc_chairs", "chairs", "sintel", "R_PWCNet-A_fine_sintel-20191218-135407", None, False],
    "S": ["PWCNet", "baseline_fine", "pwc_chairs", "chairs", "kitti", "S_PWCNet-A_fine_KITTI-20191216-125450", None, False],
    "V": ["PWCNet", "baseline_fine", "I", "things", "chairs", "V_PWCNet-I_fine_chairs-20191230-031321", None, False],
    "Y": ["PWCNet", "baseline_fine", "I", "things", "things", "Y_PWCNet-I_fine_things-20191230-024005", None, False],
    "X": ["PWCNet", "baseline_fine", "I", "things", "sintel", "X_PWCNet-I_fine_sintel-20191227-155229", None, False],
    "O": ["PWCNet", "baseline_fine", "I", "things", "kitti", "O_PWCNet-I_fine_kitti-20191226-230605", None, False],

    # baseline_single flowet
    "D": ["FlowNet1S", "baseline_single", None, "chairs", None, "D_FlowNet1S-onChairs-20191205-145310", None, True],
    "E": ["FlowNet1S", "baseline_single", None, "things", None, "E_FlowNet1S-onThings-20191205-115159", None, True],
    "M": ["FlowNet1S", "baseline_single", None, "sintel", None, "M_FlowNet1S-sintel-20191220-111613", None, True],
    "U": ["FlowNet1S", "baseline_single", None, "kitti", None, "U_FlowNet1S-kitti-20191220-110932", None, True],
    # baseline_fine flownet
    "L": ["FlowNet1S", "baseline_fine", "D", "chairs", "chairs", "L_FlowNet1S-D_fine_chairs-20191218-134755", None, False],
    "J": ["FlowNet1S", "baseline_fine", "D", "chairs", "things", "J_FlowNet1S-D_fine_things-20191216-130336", None, False],
    "N": ["FlowNet1S", "baseline_fine", "D", "chairs", "sintel", "N_FlowNet1S-D_fine_sintel-20191218-140407", None, False],
    "P": ["FlowNet1S", "baseline_fine", "D", "chairs", "kitti", "P_FlowNet1S-D_fine_kitti-20191227-005059", None, False],
    "ZZ": ["FlowNet1S", "baseline_fine", "E", "things", "chairs", "ZZ_FlowNet1S-E_fine_chairs-20200111-180519", None, False],
    "ZT": ["FlowNet1S", "baseline_fine", "E", "things", "things", "ZT_FlowNet1S-E_fine_things_res1-20200116-140705", None, False],
    "Z": ["FlowNet1S", "baseline_fine", "E", "things", "sintel", "Z_FlowNet1S-sintel-20191227-163340", None, False],
    "Q": ["FlowNet1S", "baseline_fine", "E", "things", "kitti", "Q_FlowNet1S-E_fine_kitti-20191227-020409", None, False],

    # baseline_single pwc repeated
    "C": ["PWCNet", "baseline_single_repeated", None, "chairs", None, "C_PWCNet-onChairs-20191126-113818", None, False],
    "I2": ["PWCNet", "baseline_single_repeated", None, "things", None, "I2_PWCNet-things-20191230-022450", None, False],
    "H2": ["PWCNet", "baseline_single_repeated", None, "sintel", None, "H2_PWCNet-sintel-20191227-162133", None, False],
    # baseline_single flownet repeated
    "T": ["FlowNet1S", "baseline_single_repeated", None, "chairs", None, "T_FlowNet1S-sintel-20191227-163340", None, False],

    # baseline_fine pwc repeated
    "G": ["PWCNet", "baseline_fine_repeated", "pwc_chairs", "chairs", "sintel", "G_PWCNet-A_fine_sintel-20191212-134449", None, False],

    # baseline_fine pwc special
    "NF01": ["PWCNet", "baseline_fine_special", "H", "sintel", "sintel", "NF01_PWCNet-H_fine_sintel-20191230-021221", None, False],


    # fusing blind
    "blind_ah": ["PWCNet", "fused_blind", "pwc_chairs,H", "chairs,sintel", None, "blind@ah", None, False],
    "blind_ai": ["PWCNet", "fused_blind", "pwc_chairs,I", "chairs,things", None, "blind@ai", None, False],
    "blind_aw": ["PWCNet", "fused_blind", "pwc_chairs,W", "chairs,kitti", None, "blind@aw", None, False],
    "blind_ha": ["PWCNet", "fused_blind", "H,pwc_chairs", "sintel,chairs", None, "blind@ha", None, False],
    "blind_hi": ["PWCNet", "fused_blind", "H,I", "sintel,things", None, "blind@hi", None, False],
    "blind_hw": ["PWCNet", "fused_blind", "H,W", "sintel,kitti", None, "blind@hw", None, False],
    "blind_ia": ["PWCNet", "fused_blind", "I,pwc_chairs", "things,chairs", None, "blind@ia", None, False],
    "blind_ih": ["PWCNet", "fused_blind", "I,H", "things,sintel", None, "blind@ih", None, False],
    "blind_iw": ["PWCNet", "fused_blind", "I,W", "things,kitti", None, "blind@iw", None, False],
    "blind_wa": ["PWCNet", "fused_blind", "W,pwc_chairs", "kitti,chairs", None, "blind@wa", None, False],
    "blind_wh": ["PWCNet", "fused_blind", "W,H", "kitti,sintel", None, "blind@wh", None, False],
    "blind_wi": ["PWCNet", "fused_blind", "W,I", "kitti,things", None, "blind@wi", None, False],
    # fusing blind finetuned
    "000": ["PWCNet", "fused_blind_fine", "pwc_chairs,I", "chairs,things", "sintel", "000_PWCNet-AI_fine_sintel-20191220-133049", None, False],
    "001": ["PWCNet", "fused_blind_fine", "I,pwc_chairs", "things,chairs", "sintel", "001_PWCNet-IA_fine_sintel-20191220-134913", None, False],
    "002": ["PWCNet", "fused_blind_fine", "pwc_chairs,H", "chairs,sintel", "sintel", "002_PWCNet-AH_fine_sintel-20191225-203650", None, False],
    "003": ["PWCNet", "fused_blind_fine", "H,pwc_chairs", "sintel,chairs", "sintel", "003_PWCNet-HA_fine_sintel-20191225-203955", None, False],
    "004": ["PWCNet", "fused_blind_fine", "I,H", "things,sintel", "sintel", "004_PWCNet-IH_fine_sintel-20191225-204252", None, False],
    "005": ["PWCNet", "fused_blind_fine", "H,I", "sintel,things", "sintel", "005_PWCNet-HI_fine_sintel-20191225-204931", None, False],
    # fusing conv33 not trained
    "conv33_ah": ["PWCNetConv33Fusion", "fused_conv33", "pwc_chairs,H", "chairs,sintel", None, "convBlind@ah", None, False],
    "conv33_ai": ["PWCNetConv33Fusion", "fused_conv33", "pwc_chairs,I", "chairs,things", None, "convBlind@ai", None, False],
    "conv33_aw": ["PWCNetConv33Fusion", "fused_conv33", "pwc_chairs,W", "chairs,kitti", None, "convBlind@aw", None, False],
    "conv33_ha": ["PWCNetConv33Fusion", "fused_conv33", "H,pwc_chairs", "sintel,chairs", None, "convBlind@ha", None, False],
    "conv33_hi": ["PWCNetConv33Fusion", "fused_conv33", "H,I", "sintel,things", None, "convBlind@hi", None, False],
    "conv33_hw": ["PWCNetConv33Fusion", "fused_conv33", "H,W", "sintel,kitti", None, "convBlind@hw", None, False],
    "conv33_ia": ["PWCNetConv33Fusion", "fused_conv33", "I,pwc_chairs", "things,chairs", None, "convBlind@ia", None, False],
    "conv33_ih": ["PWCNetConv33Fusion", "fused_conv33", "I,H", "things,sintel", None, "convBlind@ih", None, False],
    "conv33_iw": ["PWCNetConv33Fusion", "fused_conv33", "I,W", "things,kitti", None, "convBlind@iw", None, False],
    "conv33_wa": ["PWCNetConv33Fusion", "fused_conv33", "W,pwc_chairs", "kitti,chairs", None, "convBlind@wa", None, False],
    "conv33_wh": ["PWCNetConv33Fusion", "fused_conv33", "W,H", "kitti,sintel", None, "convBlind@wh", None, False],
    "conv33_wi": ["PWCNetConv33Fusion", "fused_conv33", "W,I", "kitti,things", None, "convBlind@wi", None, False],
    # fusing conv33 finetuned
    "0C300": ["PWCNetConv33Fusion", "fused_conv33_fine", "I,pwc_chairs", "things,chairs", "chairs", "0C300_PWCNetConv33Fusion-fine_chairs-20200105-171055", None, False],
    "0C301": ["PWCNetConv33Fusion", "fused_conv33_fine", "pwc_chairs,I", "chairs,things", "chairs", "0C301_PWCNetConv33Fusion-fine_chairs-20200105-171258", None, False],
    "0C302": ["PWCNetConv33Fusion", "fused_conv33_fine", "H,pwc_chairs", "sintel,chairs", "chairs", "0C302_PWCNetConv33Fusion-fine_chairs-20200105-171550", None, False],
    "0C303": ["PWCNetConv33Fusion", "fused_conv33_fine", "pwc_chairs,H", "chairs,sintel", "chairs", "0C303_PWCNetConv33Fusion-fine_chairs-20200105-171857", None, False],
    "0C304": ["PWCNetConv33Fusion", "fused_conv33_fine", "H,I", "sintel,things", "chairs", "0C304_PWCNetConv33Fusion-fine_chairs-20200111-174643", None, False],
    "0C305": ["PWCNetConv33Fusion", "fused_conv33_fine", "I,H", "things,sintel", "chairs", "0C305_res2_PWCNetConv33Fusion-fine_chairs-20200116-141744", None, False],
    "0C306": ["PWCNetConv33Fusion", "fused_conv33_fine", "W,pwc_chairs", "kitti,chairs", "chairs", "0C306_PWCNetConv33Fusion-fine_chairs-20200105-172316", None, False],
    "0C307": ["PWCNetConv33Fusion", "fused_conv33_fine", "pwc_chairs,W", "chairs,kitti", "chairs", "0C307_PWCNetConv33Fusion-fine_chairs-20200111-174308", None, False],
    "0C308": ["PWCNetConv33Fusion", "fused_conv33_fine", "W,I", "kitti,things", "chairs", "0C308_PWCNetConv33Fusion-fine_chairs-20200116-020142", None, False],
    "0C309": ["PWCNetConv33Fusion", "fused_conv33_fine", "I,W", "things,kitti", "chairs", "0C309_PWCNetConv33Fusion-fine_chairs-20200116-141109", None, False],
    "0C310": ["PWCNetConv33Fusion", "fused_conv33_fine", "W,H", "kitti,sintel", "chairs", "0C310_PWCNetConv33Fusion-fine_chairs-20200116-022159", None, False],
    "0C311": ["PWCNetConv33Fusion", "fused_conv33_fine", "H,W", "sintel,kitti", "chairs", "0C311_PWCNetConv33Fusion-fine_chairs-20200122-163116", None, False],

    # baseline models PWC no fine - x1 zero
    "x1ZeroBlind_A": ["PWCNetX1Zero", "x1_zero_baseline", "pwc_chairs", "chairs", None, "A_PWCNet-onChairs-20191121-171532", None, False],
    "x1ZeroBlind_I": ["PWCNetX1Zero", "x1_zero_baseline", "I", "things", None, "I_PWCNet-things_20191209-131019", None, False],
    "x1ZeroBlind_H": ["PWCNetX1Zero", "x1_zero_baseline", "H", "sintel", None, "H_PWCNet-sintel-20191209-150448", None, False],
    "x1ZeroBlind_W": ["PWCNetX1Zero", "x1_zero_baseline", "W", "kitti", None, "W_PWCNet-kitti-20191216-124247", None, False],
    # baseline models PWC finetuned - x1 zero
    "x1ZeroBlind_F": ["PWCNetX1Zero", "x1_zero_baseline_fine", "F", "chairs", "chairs", "F_PWCNet-A_fine_chairs-20191212-133136", None, False],
    "x1ZeroBlind_K": ["PWCNetX1Zero", "x1_zero_baseline_fine", "K", "chairs", "things", "K_PWCNet-A_fine_things-20191212-133436", None, False],
    "x1ZeroBlind_R": ["PWCNetX1Zero", "x1_zero_baseline_fine", "R", "chairs", "sintel", "R_PWCNet-A_fine_sintel-20191218-135407", None, False],
    "x1ZeroBlind_S": ["PWCNetX1Zero", "x1_zero_baseline_fine", "S", "chairs", "kitti", "S_PWCNet-A_fine_KITTI-20191216-125450", None, False],
    "x1ZeroBlind_V": ["PWCNetX1Zero", "x1_zero_baseline_fine", "V", "things", "chairs", "V_PWCNet-I_fine_chairs-20191230-031321", None, False],
    "x1ZeroBlind_Y": ["PWCNetX1Zero", "x1_zero_baseline_fine", "Y", "things", "things", "Y_PWCNet-I_fine_things-20191230-024005", None, False],
    "x1ZeroBlind_X": ["PWCNetX1Zero", "x1_zero_baseline_fine", "X", "things", "sintel", "X_PWCNet-I_fine_sintel-20191227-155229", None, False],
    "x1ZeroBlind_O": ["PWCNetX1Zero", "x1_zero_baseline_fine", "O", "things", "kitti", "O_PWCNet-I_fine_kitti-20191226-230605", None, False],

    # fused blind x1 zero
    "x1ZeroBlind_ah": ["PWCNetX1Zero", "x1_zero_baseline_fused", "pwc_chairs,H", "chairs,sintel", None, "blind@ah", None, False],
    "x1ZeroBlind_ai": ["PWCNetX1Zero", "x1_zero_baseline_fused", "pwc_chairs,I", "chairs,things", None, "blind@ai", None, False],
    "x1ZeroBlind_aw": ["PWCNetX1Zero", "x1_zero_baseline_fused", "pwc_chairs,W", "chairs,kitti", None, "blind@aw", None, False],
    "x1ZeroBlind_ha": ["PWCNetX1Zero", "x1_zero_baseline_fused", "H,pwc_chairs", "sintel,chairs", None, "blind@ha", None, False],
    "x1ZeroBlind_hi": ["PWCNetX1Zero", "x1_zero_baseline_fused", "H,I", "sintel,things", None, "blind@hi", None, False],
    "x1ZeroBlind_hw": ["PWCNetX1Zero", "x1_zero_baseline_fused", "H,W", "sintel,kitti", None, "blind@hw", None, False],
    "x1ZeroBlind_ia": ["PWCNetX1Zero", "x1_zero_baseline_fused", "I,pwc_chairs", "things,chairs", None, "blind@ia", None, False],
    "x1ZeroBlind_ih": ["PWCNetX1Zero", "x1_zero_baseline_fused", "I,H", "things,sintel", None, "blind@ih", None, False],
    "x1ZeroBlind_iw": ["PWCNetX1Zero", "x1_zero_baseline_fused", "I,W", "things,kitti", None, "blind@iw", None, False],
    "x1ZeroBlind_wa": ["PWCNetX1Zero", "x1_zero_baseline_fused", "W,pwc_chairs", "kitti,chairs", None, "blind@wa", None, False],
    "x1ZeroBlind_wh": ["PWCNetX1Zero", "x1_zero_baseline_fused", "W,H", "kitti,sintel", None, "blind@wh", None, False],
    "x1ZeroBlind_wi": ["PWCNetX1Zero", "x1_zero_baseline_fused", "W,I", "kitti,things", None, "blind@wi", None, False],

    # baseline pwc without X1 connection
    "pwcWOX1_chairs": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "chairs", None, "WOX1_chairs_PWCNetWOX1Connection-20200122-164023", None, True],
    "pwcWOX1_things": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "things", None, "WOX1_PWCNetWOX1Connection-things-20200127-234143", None, True],
    "pwcWOX1_sintel": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "sintel", None, "WOX1_PWCNetWOX1Connection-sintel-20200127-232828", None, True],
    "pwcWOX1_kitti": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_fixed_aug-20200225-090042", None, True],
    #"pwcWOX1_kitti_old": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "WOX1_PWCNetWOX1Connection-kitti-20200128-000101", None, False],
    #"pwcWOX1_kitti_temp": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042", None, False],
    #"pwcWOX1_kitti_tempA": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_00450.ckpt", None, False],
    #"pwcWOX1_kitti_tempB": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_01050.ckpt", None, False],
    #"pwcWOX1_kitti_tempC": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_01950.ckpt", None, False],
    #"pwcWOX1_kitti_tempD": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_03000.ckpt", None, False],
    #"pwcWOX1_kitti_tempE": ["PWCNetWOX1Connection", "without_x1_connection_baseline", None, "kitti", None, "PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_latest.ckpt", None, False],

    # PWCNetWOX1 finetuned
    "pwcWOX1_chairs_fine_things": ["PWCNetWOX1Connection", "without_x1_connection_finetuned", "pwcWOX1_chairs", "chairs", "things", "PWCNetWOX1Connection-WOX1Chairs_fine_things-20200206-153903", None, False],

    # pwc without x1 connection between encoder, decoder and context network
    "WOX1Ext_chairs": ["PWCNetWOX1ConnectionExt", "without_x1_connection_ext_baseline", None, "chairs", None, "WOX1Ext_PWCNetWOX1ConnectionExt-onChairs-20200209-181206", None, True],

    # blind fused pwc without X1 connection
    "WOX1Blind_ct": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_chairs,pwcWOX1_things", "chairs,things", None, "WOX1Blind@ct", None, False],
    "WOX1Blind_cs": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_chairs,pwcWOX1_sintel", "chairs,sintel", None, "WOX1Blind@cs", None, False],
    "WOX1Blind_ck": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_chairs,pwcWOX1_kitti", "chairs,kitti", None, "WOX1Blind@ck", None, False],
    "WOX1Blind_tc": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_things,pwcWOX1_chairs", "things,chairs", None, "WOX1Blind@tc", None, False],
    "WOX1Blind_ts": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_things,pwcWOX1_sintel", "things,sintel", None, "WOX1Blind@ts", None, False],
    "WOX1Blind_tk": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_things,pwcWOX1_kitti", "things,kitti", None, "WOX1Blind@tk", None, False],
    "WOX1Blind_sc": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_sintel,pwcWOX1_chairs", "sintel,chairs", None, "WOX1Blind@sc", None, False],
    "WOX1Blind_st": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_sintel,pwcWOX1_things", "sintel,things", None, "WOX1Blind@st", None, False],
    "WOX1Blind_sk": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_sintel,pwcWOX1_kitti", "sintel,kitti", None, "WOX1Blind@sk", None, False],
    "WOX1Blind_kc": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_kitti,pwcWOX1_chairs", "kitti,chairs", None, "WOX1Blind@kc", None, False],
    "WOX1Blind_kt": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_kitti,pwcWOX1_things", "kitti,things", None, "WOX1Blind@kt", None, False],
    "WOX1Blind_ks": ["PWCNetWOX1Connection", "without_x1_connection_blind_fused", "pwcWOX1_kitti,pwcWOX1_sintel", "kitti,sintel", None, "WOX1Blind@ks", None, False],

    # pwcWOX1 expert model split CTSK
    "expertWOX1_CTSK_split02_known": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "error", False],
    "expertWOX1_CTSK_split02_expert0": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert0", True],
    "expertWOX1_CTSK_split02_expert1": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert1", True],
    "expertWOX1_CTSK_split02_expert2": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert2", True],
    "expertWOX1_CTSK_split02_expert3": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split_02", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert3", True],

    # pwc expert model split failed kitti augmentation
    "expert_split02_failedAug_known": ["CTSKPWCExpertNet02", "pwc_expert_split_02_failed", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet_failedAugm-20200124-000701", "error", False],
    "expert_split02_failedAug_expert0": ["CTSKPWCExpertNet02", "pwc_expert_split_02_failed", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet_failedAugm-20200124-000701", "expert0", True],
    "expert_split02_failedAug_expert1": ["CTSKPWCExpertNet02", "pwc_expert_split_02_failed", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet_failedAugm-20200124-000701", "expert1", True],
    "expert_split02_failedAug_expert2": ["CTSKPWCExpertNet02", "pwc_expert_split_02_failed", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet_failedAugm-20200124-000701", "expert2", True],
    "expert_split02_failedAug_expert3": ["CTSKPWCExpertNet02", "pwc_expert_split_02_failed", None, "chairs,things,sintel,kitti", None, "expert_base02_PWCExpertNet_failedAugm-20200124-000701", "expert3", True],

    # pwcWOX1 expert model add CTSK
    "expertWOX1_CTSK_add01_known":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "error", False],
    "expertWOX1_CTSK_add01_no_expert": ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "noExpert", False],
    "expertWOX1_CTSK_add01_expert0": ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert0", True],
    "expertWOX1_CTSK_add01_expert1": ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert1", True],
    "expertWOX1_CTSK_add01_expert2": ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert2", True],
    "expertWOX1_CTSK_add01_expert3": ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert3", True],

    # pwcWOX1 expert model linAdd CTSK
    "expertWOX1_CTSK_linAdd01_known":   ["CTSKPWCExpertNetWOX1LinAdd01", "pwc_expert_linadd_01", None, "chairs,things,sintel,kitti", None, "expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307", "error", False],
    "expertWOX1_CTSK_linAdd01_no_expert": ["CTSKPWCExpertNetWOX1LinAdd01", "pwc_expert_linadd_01", None, "chairs,things,sintel,kitti", None, "expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307", "noExpert", False],
    "expertWOX1_CTSK_linAdd01_expert0": ["CTSKPWCExpertNetWOX1LinAdd01", "pwc_expert_linadd_01", None, "chairs,things,sintel,kitti", None, "expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307", "expert0", True],
    "expertWOX1_CTSK_linAdd01_expert1": ["CTSKPWCExpertNetWOX1LinAdd01", "pwc_expert_linadd_01", None, "chairs,things,sintel,kitti", None, "expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307", "expert1", True],
    "expertWOX1_CTSK_linAdd01_expert2": ["CTSKPWCExpertNetWOX1LinAdd01", "pwc_expert_linadd_01", None, "chairs,things,sintel,kitti", None, "expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307", "expert2", True],
    "expertWOX1_CTSK_linAdd01_expert3": ["CTSKPWCExpertNetWOX1LinAdd01", "pwc_expert_linadd_01", None, "chairs,things,sintel,kitti", None, "expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307", "expert3", True],

    # pwcWOX1 expert model add CTS
    "expertWOX1_CTS_add01_no_expert": ["CTSPWCExpertNetWOX1Add01", "pwc_expert_add_01_CTS", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331", "noExpert", False],
    "expertWOX1_CTS_add01_expert0": ["CTSPWCExpertNetWOX1Add01", "pwc_expert_add_01_CTS", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331", "expert0", True],
    "expertWOX1_CTS_add01_expert1": ["CTSPWCExpertNetWOX1Add01", "pwc_expert_add_01_CTS", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331", "expert1", True],
    "expertWOX1_CTS_add01_expert2": ["CTSPWCExpertNetWOX1Add01", "pwc_expert_add_01_CTS", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331", "expert2", True],

    # pwc expert model add failed kitti augmentation
    "expert_add01_failedAug_known":   ["CTSKPWCExpertNetAdd01_failed", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "error", False],
    "expert_add01_failedAug_no_expert": ["CTSKPWCExpertNetAdd01_failed", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "noExpert", False],
    "expert_add01_failedAug_expert0": ["CTSKPWCExpertNetAdd01_failed", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert0", True],
    "expert_add01_failedAug_expert1": ["CTSKPWCExpertNetAdd01_failed", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert1", True],
    "expert_add01_failedAug_expert2": ["CTSKPWCExpertNetAdd01_failed", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert2", True],
    "expert_add01_failedAug_expert3": ["CTSKPWCExpertNetAdd01_failed", "pwc_expert_add_01", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNet-20200124-174956", "expert3", True],

    # pwc expert model CTS add
    #TODO "expert_CTS_add01_known": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "error", False],
    "expert_CTS_add01_no_expert": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "noExpert", False],
    "expert_CTS_add01_expert0": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert0", True],
    "expert_CTS_add01_expert1": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert1", True],
    "expert_CTS_add01_expert2": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert2", True],

    # pwcWOX1 expert model add encoder only CTSK
    "expertWOX1_encoder_only_CTSK_add01_known":   ["CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly", "pwc_expert_add_01_encoder_only", None, "chairs,things,sintel,kitti", None, "expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437", "error", False],
    "expertWOX1_encoder_only_CTSK_add01_no_expert": ["CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly", "pwc_expert_add_01_encoder_only", None, "chairs,things,sintel,kitti", None, "expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437", "noExpert", False],
    "expertWOX1_encoder_only_CTSK_add01_expert0": ["CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly", "pwc_expert_add_01_encoder_only", None, "chairs,things,sintel,kitti", None, "expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437", "expert0", True],
    "expertWOX1_encoder_only_CTSK_add01_expert1": ["CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly", "pwc_expert_add_01_encoder_only", None, "chairs,things,sintel,kitti", None, "expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437", "expert1", True],
    "expertWOX1_encoder_only_CTSK_add01_expert2": ["CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly", "pwc_expert_add_01_encoder_only", None, "chairs,things,sintel,kitti", None, "expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437", "expert2", True],
    "expertWOX1_encoder_only_CTSK_add01_expert3": ["CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly", "pwc_expert_add_01_encoder_only", None, "chairs,things,sintel,kitti", None, "expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437", "expert3", True],

    # pwcWOX1 expert model split trace
    "expertWOX1_CTSK_split02_known_iter01": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_001.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter10": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_010.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter20": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_020.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter30": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_030.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter40": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_040.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter50": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_050.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter52": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_052.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter54": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_054.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter56": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_056.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter58": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_058.ckpt", "error", False],
    "expertWOX1_CTSK_split02_known_iter60": ["CTSKPWCExpertNet02WOX1", "pwc_expert_split02_trace", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_060.ckpt", "error", False],

    # pwcWOX1 expert model add trace
    "expertWOX1_CTSK_add01_known_iter01":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_001.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter10":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_010.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter20":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_020.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter30":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_030.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter40":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_040.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter50":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_050.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter52":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_052.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter54":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_054.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter56":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_056.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter58":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_058.ckpt", "error", False],
    "expertWOX1_CTSK_add01_known_iter60":   ["CTSKPWCExpertNetWOX1Add01", "pwc_expert_add01_trace", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_060.ckpt", "error", False],

    # CTSK WOX1 trace
    "pwcWOX1_on_CTSK_iter01": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_001.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter10": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_010.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter20": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_020.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter30": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_030.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter40": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_040.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter50": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_050.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter52": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_052.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter54": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_054.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter56": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_056.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter58": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_058.ckpt", None, False],
    "pwcWOX1_on_CTSK_iter60": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_060.ckpt", None, False],

    # pwc trained on multiple datasets
    "pwc_on_CTSK_failedaugment": ["PWCNet", "pwc_on_multiple", None, "chairs,things,sintel,kitti", None, "expert__noExpert_failedaugment_PWCNet-20200127-234847", None, False],
    "pwc_on_CTSK": ["PWCNet", "pwc_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_noExpert_PWCNet-20200225-093211", None, True],
    # pwcWOX1 trained on multiple datasets
    "pwcWOX1_on_CTSK": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel,kitti", None, "expert_base_wox1_PWCNetWOX1Connection-20200309-153241", None, True],
    "pwcWOX1_on_CTS": ["PWCNetWOX1Connection", "pwcWOX1_on_multiple", None, "chairs,things,sintel", None, "pwc_noExperts_CTS_wox1_PWCNetWOX1Connection-20200311-213609", None, True],

    # pwc iteration inspection
    "pwc_chairs_iter_148": ["PWCNet", "pwc_wox1_iteration_inspection", None, "chairs", None, "iter_PWCNet-chairs_148/checkpoint_iter_148.ckpt", None, False],

    # pwc down sampling encoder
    "pwcDSEncoder_chairs": ["PWCNetDSEncoder", "pwc_ds_encoder", None, "chairs", None, "PWC_DSEncoder_PWCNetDSEncoder-onChairs-20200209-190403", None, True],

    # pwc residual flow
    "pwc_residual_flow_CTSK": ["PWCNetResidualFlow", "pwc_residual_flow", None, "chairs,things,sintel,kitti", None, "pwc_residual_flow_CTSK_PWCNetResidualFlow-20200326-131601", None, True],

    # pwc every chairs
    "pwc_every_chairs_001": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_001.ckpt", None, False],
    "pwc_every_chairs_011": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_011.ckpt", None, False],
    "pwc_every_chairs_021": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_021.ckpt", None, False],
    "pwc_every_chairs_031": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_031.ckpt", None, False],
    "pwc_every_chairs_041": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_041.ckpt", None, False],
    "pwc_every_chairs_051": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_051.ckpt", None, False],
    "pwc_every_chairs_061": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_061.ckpt", None, False],
    "pwc_every_chairs_071": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_071.ckpt", None, False],
    "pwc_every_chairs_081": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_081.ckpt", None, False],
    "pwc_every_chairs_091": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_091.ckpt", None, False],
    "pwc_every_chairs_101": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_101.ckpt", None, False],
    "pwc_every_chairs_111": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_111.ckpt", None, False],
    "pwc_every_chairs_121": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_121.ckpt", None, False],
    "pwc_every_chairs_131": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_131.ckpt", None, False],
    "pwc_every_chairs_141": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_141.ckpt", None, False],
    "pwc_every_chairs_151": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_151.ckpt", None, False],
    "pwc_every_chairs_161": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_161.ckpt", None, False],
    "pwc_every_chairs_171": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_171.ckpt", None, False],
    "pwc_every_chairs_181": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_181.ckpt", None, False],
    "pwc_every_chairs_191": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_191.ckpt", None, False],
    "pwc_every_chairs_201": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_201.ckpt", None, False],
    "pwc_every_chairs_209": ["PWCNet", "pwc_progress_tracing", None, "chairs", None, "PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_209.ckpt", None, False],

    # pwcWOX1 every chairs
    "pwcWOX1_every_chairs_001": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_001.ckpt", None, False],
    "pwcWOX1_every_chairs_011": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_011.ckpt", None, False],
    "pwcWOX1_every_chairs_021": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_021.ckpt", None, False],
    "pwcWOX1_every_chairs_031": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_031.ckpt", None, False],
    "pwcWOX1_every_chairs_041": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_041.ckpt", None, False],
    "pwcWOX1_every_chairs_051": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_051.ckpt", None, False],
    "pwcWOX1_every_chairs_061": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_061.ckpt", None, False],
    "pwcWOX1_every_chairs_071": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_071.ckpt", None, False],
    "pwcWOX1_every_chairs_081": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_081.ckpt", None, False],
    "pwcWOX1_every_chairs_091": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_091.ckpt", None, False],
    "pwcWOX1_every_chairs_101": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_101.ckpt", None, False],
    "pwcWOX1_every_chairs_111": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_111.ckpt", None, False],
    "pwcWOX1_every_chairs_121": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_121.ckpt", None, False],
    "pwcWOX1_every_chairs_131": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_131.ckpt", None, False],
    "pwcWOX1_every_chairs_141": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_141.ckpt", None, False],
    "pwcWOX1_every_chairs_151": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_151.ckpt", None, False],
    "pwcWOX1_every_chairs_161": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_161.ckpt", None, False],
    "pwcWOX1_every_chairs_171": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_171.ckpt", None, False],
    "pwcWOX1_every_chairs_181": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_181.ckpt", None, False],
    "pwcWOX1_every_chairs_191": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_191.ckpt", None, False],
    "pwcWOX1_every_chairs_201": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_201.ckpt", None, False],
    "pwcWOX1_every_chairs_209": ["PWCNetWOX1Connection", "pwcWOX1_progress_tracing", None, "chairs", None, "PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_209.ckpt", None, False],

    # CTS Expert model merging
    "expert_CTS_add01_CC": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert00", True],
    "expert_CTS_add01_CT": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert01", False],
    "expert_CTS_add01_CS": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert02", False],
    "expert_CTS_add01_TC": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert10", False],
    "expert_CTS_add01_TT": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert11", True],
    "expert_CTS_add01_TS": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert12", False],
    "expert_CTS_add01_SC": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert20", False],
    "expert_CTS_add01_ST": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert21", False],
    "expert_CTS_add01_SS": ["CTSPWCExpertNetAdd01", "pwc_expert_CTS_add_01_merged", None, "chairs,things,sintel", None, "expert_add01_CTS_PWCExpertAddNet-20200210-223344", "expert22", True],

    # CTKS WOX1 fused expert split models
    "expertWOX1_CTSK_split02_CC": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert00", True],
    "expertWOX1_CTSK_split02_CT": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert01", False],
    "expertWOX1_CTSK_split02_CS": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert02", False],
    "expertWOX1_CTSK_split02_CK": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert03", False],
    "expertWOX1_CTSK_split02_TC": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert10", False],
    "expertWOX1_CTSK_split02_TT": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert11", True],
    "expertWOX1_CTSK_split02_TS": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert12", False],
    "expertWOX1_CTSK_split02_TK": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert13", False],
    "expertWOX1_CTSK_split02_SC": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert20", False],
    "expertWOX1_CTSK_split02_ST": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert21", False],
    "expertWOX1_CTSK_split02_SS": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert22", True],
    "expertWOX1_CTSK_split02_SK": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert23", False],
    "expertWOX1_CTSK_split02_KC": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert30", False],
    "expertWOX1_CTSK_split02_KT": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert31", False],
    "expertWOX1_CTSK_split02_KS": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert32", False],
    "expertWOX1_CTSK_split02_KK": ["CTSKPWCExpertNet02WOX1", "pwc_expert_CTSK_split_02_merged", None, "chairs,things,sintel,kitti", None, "expert_base_PWCExpertNetWOX1-20200227-012926", "expert33", True],

    # CTKS WOX1 fused expert split models
    "expertWOX1_CTSK_add01_CC": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert00", True],
    "expertWOX1_CTSK_add01_CT": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert01", False],
    "expertWOX1_CTSK_add01_CS": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert02", False],
    "expertWOX1_CTSK_add01_CK": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert03", False],
    "expertWOX1_CTSK_add01_TC": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert10", False],
    "expertWOX1_CTSK_add01_TT": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert11", True],
    "expertWOX1_CTSK_add01_TS": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert12", False],
    "expertWOX1_CTSK_add01_TK": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert13", False],
    "expertWOX1_CTSK_add01_SC": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert20", False],
    "expertWOX1_CTSK_add01_ST": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert21", False],
    "expertWOX1_CTSK_add01_SS": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert22", True],
    "expertWOX1_CTSK_add01_SK": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert23", False],
    "expertWOX1_CTSK_add01_KC": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert30", False],
    "expertWOX1_CTSK_add01_KT": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert31", False],
    "expertWOX1_CTSK_add01_KS": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert32", False],
    "expertWOX1_CTSK_add01_KK": ["CTSKPWCExpertNet01WOX1Add", "pwc_expert_CTSK_add_01_merged", None, "chairs,things,sintel,kitti", None, "expert_add01_PWCExpertAddNetWOX1-20200227-013322", "expert33", True],
}

model_meta_ordering = [
    # baseline_single pwc
    "pwc_chairs", "I", "H", "pwc_kitti",
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
    "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti",
    # old kitti results
    #"pwcWOX1_kitti_temp", "W", "pwc_kitti_temp",
    #"pwcWOX1_kitti_tempA", "pwcWOX1_kitti_tempB", "pwcWOX1_kitti_tempC", "pwcWOX1_kitti_tempD", "pwcWOX1_kitti_tempE",
    # PWCNetWOX1 finetuned
    "pwcWOX1_chairs_fine_things",
    # pwc without x1 connection between encoder, decoder and context network
    "WOX1Ext_chairs",
    # blind fused pwc without X1 connection
    "WOX1Blind_ct", "WOX1Blind_cs", "WOX1Blind_ck", "WOX1Blind_tc", "WOX1Blind_ts", "WOX1Blind_tk",
    "WOX1Blind_sc", "WOX1Blind_st", "WOX1Blind_sk", "WOX1Blind_kc", "WOX1Blind_kt", "WOX1Blind_ks",
    # pwc expert model split
    "expertWOX1_CTSK_split02_known",
    "expertWOX1_CTSK_split02_expert0", "expertWOX1_CTSK_split02_expert1", "expertWOX1_CTSK_split02_expert2",
    "expertWOX1_CTSK_split02_expert3",
    # pwc expert model split failed kitti augmentation
    "expert_split02_failedAug_known",
    "expert_split02_failedAug_expert0", "expert_split02_failedAug_expert1", "expert_split02_failedAug_expert2",
    "expert_split02_failedAug_expert3",
    # pwc expert model add failed
    "expert_add01_failedAug_known", "expert_add01_failedAug_no_expert",
    "expert_add01_failedAug_expert0", "expert_add01_failedAug_expert1", "expert_add01_failedAug_expert2",
    "expert_add01_failedAug_expert3",
    # pwc expert model add wox1
    "expertWOX1_CTSK_add01_known", "expertWOX1_CTSK_add01_no_expert",
    "expertWOX1_CTSK_add01_expert0", "expertWOX1_CTSK_add01_expert1", "expertWOX1_CTSK_add01_expert2",
    "expertWOX1_CTSK_add01_expert3",
    # pwc expert model linAdd wox1
    "expertWOX1_CTSK_linAdd01_known", "expertWOX1_CTSK_linAdd01_no_expert",
    "expertWOX1_CTSK_linAdd01_expert0", "expertWOX1_CTSK_linAdd01_expert1", "expertWOX1_CTSK_linAdd01_expert2",
    "expertWOX1_CTSK_linAdd01_expert3",
    # pwc expert model add wox1 CTS
    "expertWOX1_CTS_add01_no_expert",
    "expertWOX1_CTS_add01_expert0", "expertWOX1_CTS_add01_expert1", "expertWOX1_CTS_add01_expert2",
    # pwc expert model CTS add
    "expert_CTS_add01_no_expert",
    "expert_CTS_add01_expert0", "expert_CTS_add01_expert1", "expert_CTS_add01_expert2",
    # pwcWOX1 expert model add encoder only CTSK
    "expertWOX1_encoder_only_CTSK_add01_known", "expertWOX1_encoder_only_CTSK_add01_no_expert",
    "expertWOX1_encoder_only_CTSK_add01_expert0", "expertWOX1_encoder_only_CTSK_add01_expert1",
    "expertWOX1_encoder_only_CTSK_add01_expert2", "expertWOX1_encoder_only_CTSK_add01_expert3",
    # pwcWOX1 expert model split trace
    "expertWOX1_CTSK_split02_known_iter01", "expertWOX1_CTSK_split02_known_iter10", "expertWOX1_CTSK_split02_known_iter20",
    "expertWOX1_CTSK_split02_known_iter30", "expertWOX1_CTSK_split02_known_iter40", "expertWOX1_CTSK_split02_known_iter50",
    "expertWOX1_CTSK_split02_known_iter52", "expertWOX1_CTSK_split02_known_iter54", "expertWOX1_CTSK_split02_known_iter56",
    "expertWOX1_CTSK_split02_known_iter58", "expertWOX1_CTSK_split02_known_iter60",

    # pwcWOX1 expert model add trace
    "expertWOX1_CTSK_add01_known_iter01", "expertWOX1_CTSK_add01_known_iter10", "expertWOX1_CTSK_add01_known_iter20",
    "expertWOX1_CTSK_add01_known_iter30", "expertWOX1_CTSK_add01_known_iter40", "expertWOX1_CTSK_add01_known_iter50",
    "expertWOX1_CTSK_add01_known_iter52", "expertWOX1_CTSK_add01_known_iter54", "expertWOX1_CTSK_add01_known_iter56",
    "expertWOX1_CTSK_add01_known_iter58", "expertWOX1_CTSK_add01_known_iter60",

    # CTSK WOX1 trace
    "pwcWOX1_on_CTSK_iter01", "pwcWOX1_on_CTSK_iter10", "pwcWOX1_on_CTSK_iter20", "pwcWOX1_on_CTSK_iter30",
    "pwcWOX1_on_CTSK_iter40", "pwcWOX1_on_CTSK_iter50", "pwcWOX1_on_CTSK_iter52", "pwcWOX1_on_CTSK_iter54",
    "pwcWOX1_on_CTSK_iter56", "pwcWOX1_on_CTSK_iter58", "pwcWOX1_on_CTSK_iter60",

    # pwc trained on multiple datasets
    "pwc_on_CTSK_failedaugment",
    "pwc_on_CTSK",
    # pwcWOX1 trained on multiple datasets
    "pwcWOX1_on_CTSK", "pwcWOX1_on_CTS",
    # pwc iteration inspection
    "pwc_chairs_iter_148",
    # pwc down sampling encoder
    "pwcDSEncoder_chairs",
    # pwc residual flow
    "pwc_residual_flow_CTSK",

    # pwc every chairs
    "pwc_every_chairs_001", "pwc_every_chairs_011", "pwc_every_chairs_021", "pwc_every_chairs_031",
    "pwc_every_chairs_041", "pwc_every_chairs_051", "pwc_every_chairs_061", "pwc_every_chairs_071",
    "pwc_every_chairs_081", "pwc_every_chairs_091", "pwc_every_chairs_101", "pwc_every_chairs_111",
    "pwc_every_chairs_121", "pwc_every_chairs_131", "pwc_every_chairs_141", "pwc_every_chairs_151",
    "pwc_every_chairs_161", "pwc_every_chairs_171", "pwc_every_chairs_181", "pwc_every_chairs_191",
    "pwc_every_chairs_201", "pwc_every_chairs_209",
    # pwcWOX1 every chairs
    "pwcWOX1_every_chairs_001", "pwcWOX1_every_chairs_011", "pwcWOX1_every_chairs_021", "pwcWOX1_every_chairs_031",
    "pwcWOX1_every_chairs_041", "pwcWOX1_every_chairs_051", "pwcWOX1_every_chairs_061", "pwcWOX1_every_chairs_071",
    "pwcWOX1_every_chairs_081", "pwcWOX1_every_chairs_091", "pwcWOX1_every_chairs_101", "pwcWOX1_every_chairs_111",
    "pwcWOX1_every_chairs_121", "pwcWOX1_every_chairs_131", "pwcWOX1_every_chairs_141", "pwcWOX1_every_chairs_151",
    "pwcWOX1_every_chairs_161", "pwcWOX1_every_chairs_171", "pwcWOX1_every_chairs_181", "pwcWOX1_every_chairs_191",
    "pwcWOX1_every_chairs_201", "pwcWOX1_every_chairs_209",
    # CTS Expert model merging
    "expert_CTS_add01_CC", "expert_CTS_add01_CT", "expert_CTS_add01_CS",
    "expert_CTS_add01_TC", "expert_CTS_add01_TT", "expert_CTS_add01_TS",
    "expert_CTS_add01_SC", "expert_CTS_add01_ST", "expert_CTS_add01_SS",
    
    # CTKS WOX1 fused expert split models
    "expertWOX1_CTSK_split02_CC", "expertWOX1_CTSK_split02_CT", "expertWOX1_CTSK_split02_CS", "expertWOX1_CTSK_split02_CK",
    "expertWOX1_CTSK_split02_TC", "expertWOX1_CTSK_split02_TT", "expertWOX1_CTSK_split02_TS", "expertWOX1_CTSK_split02_TK",
    "expertWOX1_CTSK_split02_SC", "expertWOX1_CTSK_split02_ST", "expertWOX1_CTSK_split02_SS", "expertWOX1_CTSK_split02_SK",
    "expertWOX1_CTSK_split02_KC", "expertWOX1_CTSK_split02_KT", "expertWOX1_CTSK_split02_KS", "expertWOX1_CTSK_split02_KK",

    # CTKS WOX1 fused expert split models
    "expertWOX1_CTSK_add01_CC", "expertWOX1_CTSK_add01_CT", "expertWOX1_CTSK_add01_CS", "expertWOX1_CTSK_add01_CK",
    "expertWOX1_CTSK_add01_TC", "expertWOX1_CTSK_add01_TT", "expertWOX1_CTSK_add01_TS", "expertWOX1_CTSK_add01_TK",
    "expertWOX1_CTSK_add01_SC", "expertWOX1_CTSK_add01_ST", "expertWOX1_CTSK_add01_SS", "expertWOX1_CTSK_add01_SK",
    "expertWOX1_CTSK_add01_KC", "expertWOX1_CTSK_add01_KT", "expertWOX1_CTSK_add01_KS", "expertWOX1_CTSK_add01_KK"
]

model_folders = {
    "_default": "/data/dataB/models/",
    "blind": "/data/dataB/fusedModels_blind/",
    "convBlind": "/data/dataB/fusedModelsConv33/",
    "WOX1Blind": "/data/dataB/fusedModelsWOX1Conn_blind/"
}

set_diff = set(iter(model_meta.keys())).difference(model_meta_ordering)
if len(set_diff) != 0:
    print("model_meta and model_meta_ordering do not match. difference in datasets:", set_diff)
assert(len(set(iter(model_meta.keys())).difference(model_meta_ordering)) == 0)
