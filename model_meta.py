
model_meta_fields = ["model", "experiment", "base_model", "dataset_base", "dataset_fine"]
model_meta = {
    # baseline_single pwc
    "A": ["pwc", "baseline_single", None, "chairs", None],
    "I": ["pwc", "baseline_single", None, "things", None],
    "H": ["pwc", "baseline_single", None, "sintel", None],
    "W": ["pwc", "baseline_single", None, "kitti", None],
    # baseline_fine pwc
    "F": ["pwc", "baseline_fine", "A", "chairs", "chairs"],
    "K": ["pwc", "baseline_fine", "A", "chairs", "things"],
    "R": ["pwc", "baseline_fine", "A", "chairs", "sintel"],
    "S": ["pwc", "baseline_fine", "A", "chairs", "kitti"],
    "V": ["pwc", "baseline_fine", "I", "things", "chairs"],
    "Y": ["pwc", "baseline_fine", "I", "things", "things"],
    "X": ["pwc", "baseline_fine", "I", "things", "sintel"],
    "O": ["pwc", "baseline_fine", "I", "things", "kitti"],

    # baseline_single flowet
    "D": ["flownet", "baseline_single", None, "chairs", None],
    "E": ["flownet", "baseline_single", None, "things", None],
    "M": ["flownet", "baseline_single", None, "sintel", None],
    "U": ["flownet", "baseline_single", None, "kitti", None],
    # baseline_fine flownet
    "L": ["flownet", "baseline_fine", "D", "chairs", "chairs"],
    "J": ["flownet", "baseline_fine", "D", "chairs", "things"],
    "N": ["flownet", "baseline_fine", "D", "chairs", "sintel"],
    "P": ["flownet", "baseline_fine", "D", "chairs", "kitti"],
    "ZZ": ["flownet", "baseline_fine", "E", "things", "chairs"],
    "ZT": ["flownet", "baseline_fine", "E", "things", "things"],
    "Z": ["flownet", "baseline_fine", "E", "things", "sintel"],
    "Q": ["flownet", "baseline_fine", "E", "things", "kitti"],

    # baseline_single pwc repeated
    "C": ["pwc", "baseline_single_repeated", None, "chairs", None],
    "I2": ["pwc", "baseline_single_repeated", None, "things", None],
    "H2": ["pwc", "baseline_single_repeated", None, "sintel", None],
    # baseline_single flownet repeated
    "T": ["flownet", "baseline_single_repeated", None, "chairs", None],

    # baseline_fine pwc repeated
    "G": ["pwc", "baseline_fine_repeated", "A", "chairs", "sintel"],

    # baseline_fine pwc special
    "NF01": ["pwc", "baseline_fine_special", "H", "sintel", "sintel"],


    # fusing blind
    "blind_ah": ["pwc", "fused_blind", "A,H", "chairs,sintel", None],
    "blind_ai": ["pwc", "fused_blind", "A,I", "chairs,things", None],
    "blind_aw": ["pwc", "fused_blind", "A,W", "chairs,kitti", None],
    "blind_ha": ["pwc", "fused_blind", "H,A", "sintel,chairs", None],
    "blind_hi": ["pwc", "fused_blind", "H,I", "sintel,things", None],
    "blind_hw": ["pwc", "fused_blind", "H,W", "sintel,kitti", None],
    "blind_ia": ["pwc", "fused_blind", "I,A", "things,chairs", None],
    "blind_ih": ["pwc", "fused_blind", "I,H", "things,sintel", None],
    "blind_iw": ["pwc", "fused_blind", "I,W", "things,kitti", None],
    "blind_wa": ["pwc", "fused_blind", "W,A", "kitti,chairs", None],
    "blind_wh": ["pwc", "fused_blind", "W,H", "kitti,sintel", None],
    "blind_wi": ["pwc", "fused_blind", "W,I", "kitti,things", None],
    # fusing blind finetuned
    "000": ["pwc", "fused_blind_fine", "A,I", "chairs,things", "sintel"],
    "001": ["pwc", "fused_blind_fine", "I,A", "things,chairs", "sintel"],
    "002": ["pwc", "fused_blind_fine", "A,H", "chairs,sintel", "sintel"],
    "003": ["pwc", "fused_blind_fine", "H,A", "sintel,chairs", "sintel"],
    "004": ["pwc", "fused_blind_fine", "I,H", "things,sintel", "sintel"],
    "005": ["pwc", "fused_blind_fine", "H,I", "sintel,things", "sintel"],
    # fusing conv33 not trained
    "conv33_ah": ["pwcConv33", "fused_conv33", "A,H", "chairs,sintel", None],
    "conv33_ai": ["pwcConv33", "fused_conv33", "A,I", "chairs,things", None],
    "conv33_aw": ["pwcConv33", "fused_conv33", "A,W", "chairs,kitti", None],
    "conv33_ha": ["pwcConv33", "fused_conv33", "H,A", "sintel,chairs", None],
    "conv33_hi": ["pwcConv33", "fused_conv33", "H,I", "sintel,things", None],
    "conv33_hw": ["pwcConv33", "fused_conv33", "H,W", "sintel,kitti", None],
    "conv33_ia": ["pwcConv33", "fused_conv33", "I,A", "things,chairs", None],
    "conv33_ih": ["pwcConv33", "fused_conv33", "I,H", "things,sintel", None],
    "conv33_iw": ["pwcConv33", "fused_conv33", "I,W", "things,kitti", None],
    "conv33_wa": ["pwcConv33", "fused_conv33", "W,A", "kitti,chairs", None],
    "conv33_wh": ["pwcConv33", "fused_conv33", "W,H", "kitti,sintel", None],
    "conv33_wi": ["pwcConv33", "fused_conv33", "W,I", "kitti,things", None],
    # fusing conv33 finetuned
    "0C300": ["pwcConv33", "fused_conv33_fine", "I,A", "things,chairs", "chairs"],
    "0C301": ["pwcConv33", "fused_conv33_fine", "A,I", "chairs,things", "chairs"],
    "0C302": ["pwcConv33", "fused_conv33_fine", "H,A", "sintel,chairs", "chairs"],
    "0C303": ["pwcConv33", "fused_conv33_fine", "A,H", "chairs,sintel", "chairs"],
    "0C304": ["pwcConv33", "fused_conv33_fine", "H,I", "sintel,things", "chairs"],
    "0C305": ["pwcConv33", "fused_conv33_fine", "I,H", "things,sintel", "chairs"],
    "0C306": ["pwcConv33", "fused_conv33_fine", "W,A", "kitti,chairs", "chairs"],
    "0C307": ["pwcConv33", "fused_conv33_fine", "A,W", "chairs,kitti", "chairs"],
    "0C308": ["pwcConv33", "fused_conv33_fine", "W,I", "kitti,things", "chairs"],
    "0C309": ["pwcConv33", "fused_conv33_fine", "I,W", "things,kitti", "chairs"],
    "0C310": ["pwcConv33", "fused_conv33_fine", "W,H", "kitti,sintel", "chairs"],
    "0C311": ["pwcConv33", "fused_conv33_fine", "H,W", "sintel,kitti", "chairs"],

    # baseline models PWC no fine - x1 zero
    "x1ZeroBlind_A": ["pwcX1Zero", "x1_zero_baseline", "A", "chairs", None],
    "x1ZeroBlind_I": ["pwcX1Zero", "x1_zero_baseline", "I", "things", None],
    "x1ZeroBlind_H": ["pwcX1Zero", "x1_zero_baseline", "H", "sintel", None],
    "x1ZeroBlind_W": ["pwcX1Zero", "x1_zero_baseline", "W", "kitti", None],
    # baseline models PWC finetuned - x1 zero
    "x1ZeroBlind_F": ["pwcX1Zero", "x1_zero_baseline_fine", "F", "chairs", "chairs"],
    "x1ZeroBlind_K": ["pwcX1Zero", "x1_zero_baseline_fine", "K", "chairs", "things"],
    "x1ZeroBlind_R": ["pwcX1Zero", "x1_zero_baseline_fine", "R", "chairs", "sintel"],
    "x1ZeroBlind_S": ["pwcX1Zero", "x1_zero_baseline_fine", "S", "chairs", "kitti"],
    "x1ZeroBlind_V": ["pwcX1Zero", "x1_zero_baseline_fine", "V", "things", "chairs"],
    "x1ZeroBlind_Y": ["pwcX1Zero", "x1_zero_baseline_fine", "Y", "things", "things"],
    "x1ZeroBlind_X": ["pwcX1Zero", "x1_zero_baseline_fine", "X", "things", "sintel"],
    "x1ZeroBlind_O": ["pwcX1Zero", "x1_zero_baseline_fine", "O", "things", "kitti"],

    # fused blind x1 zero
    "x1ZeroBlind_ah": ["pwcX1Zero", "x1_zero_baseline_fused", "A,H", "chairs,sintel", None],
    "x1ZeroBlind_ai": ["pwcX1Zero", "x1_zero_baseline_fused", "A,I", "chairs,things", None],
    "x1ZeroBlind_aw": ["pwcX1Zero", "x1_zero_baseline_fused", "A,W", "chairs,kitti", None],
    "x1ZeroBlind_ha": ["pwcX1Zero", "x1_zero_baseline_fused", "H,A", "sintel,chairs", None],
    "x1ZeroBlind_hi": ["pwcX1Zero", "x1_zero_baseline_fused", "H,I", "sintel,things", None],
    "x1ZeroBlind_hw": ["pwcX1Zero", "x1_zero_baseline_fused", "H,W", "sintel,kitti", None],
    "x1ZeroBlind_ia": ["pwcX1Zero", "x1_zero_baseline_fused", "I,A", "things,chairs", None],
    "x1ZeroBlind_ih": ["pwcX1Zero", "x1_zero_baseline_fused", "I,H", "things,sintel", None],
    "x1ZeroBlind_iw": ["pwcX1Zero", "x1_zero_baseline_fused", "I,W", "things,kitti", None],
    "x1ZeroBlind_wa": ["pwcX1Zero", "x1_zero_baseline_fused", "W,A", "kitti,chairs", None],
    "x1ZeroBlind_wh": ["pwcX1Zero", "x1_zero_baseline_fused", "W,H", "kitti,sintel", None],
    "x1ZeroBlind_wi": ["pwcX1Zero", "x1_zero_baseline_fused", "W,I", "kitti,things", None],

    # baseline pwc without X1 connection
    "pwcWOX1_chairs": ["WOX1Connection", "without_x1_connection_baseline", None, "chairs", None],
}

model_meta_ordering = [
    # baseline_single pwc
    "A", "I", "H", "W",
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
    "pwcWOX1_chairs"
]

# print(set(iter(model_meta.keys())).difference(model_meta_ordering))
assert(len(model_meta) == len(model_meta_ordering))
