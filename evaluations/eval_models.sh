export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# base experiments
python ./evaluate_for_all_datasets.py /data/dataB/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/A.json
#x python ./evaluate_for_all_datasets.py /data/dataB/models/B_things_PWCNet-20191122-152857_incomplete/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/B.json
python ./evaluate_for_all_datasets.py /data/dataB/models/D_FlowNet1S-onChairs-20191205-145310/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/D.json
python ./evaluate_for_all_datasets.py /data/dataB/models/E_FlowNet1S-onThings-20191205-115159/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/E.json
python ./evaluate_for_all_datasets.py /data/dataB/models/F_PWCNet-A_fine_chairs-20191212-133136/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/F.json
python ./evaluate_for_all_datasets.py /data/dataB/models/H_PWCNet-sintel-20191209-150448/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/H.json
python ./evaluate_for_all_datasets.py /data/dataB/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/I.json
python ./evaluate_for_all_datasets.py /data/dataB/models/J_FlowNet1S-D_fine_things-20191216-130336/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/J.json
python ./evaluate_for_all_datasets.py /data/dataB/models/K_PWCNet-A_fine_things-20191212-133436/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/K.json
python ./evaluate_for_all_datasets.py /data/dataB/models/L_FlowNet1S-D_fine_chairs-20191218-134755/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/L.json
python ./evaluate_for_all_datasets.py /data/dataB/models/M_FlowNet1S-sintel-20191220-111613/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/M.json
python ./evaluate_for_all_datasets.py /data/dataB/models/N_FlowNet1S-D_fine_sintel-20191218-140407/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/N.json
python ./evaluate_for_all_datasets.py /data/dataB/models/O_PWCNet-I_fine_kitti-20191226-230605/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/O.json
python ./evaluate_for_all_datasets.py /data/dataB/models/P_FlowNet1S-D_fine_kitti-20191227-005059/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/P.json
python ./evaluate_for_all_datasets.py /data/dataB/models/Q_FlowNet1S-E_fine_kitti-20191227-020409/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/Q.json
python ./evaluate_for_all_datasets.py /data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/R.json
python ./evaluate_for_all_datasets.py /data/dataB/models/S_PWCNet-A_fine_KITTI-20191216-125450/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/S.json
python ./evaluate_for_all_datasets.py /data/dataB/models/T_FlowNet1S-sintel-20191227-163340/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/T.json
python ./evaluate_for_all_datasets.py /data/dataB/models/U_FlowNet1S-kitti-20191220-110932/checkpoint_best.ckpt flownet /data/dataB/meta/full_evals/U.json
python ./evaluate_for_all_datasets.py /data/dataB/models/V_PWCNet-I_fine_chairs-20191230-031321/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/V.json
python ./evaluate_for_all_datasets.py /data/dataB/models/W_PWCNet-kitti-20191216-124247/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/W.json
python ./evaluate_for_all_datasets.py /data/dataB/models/X_PWCNet-I_fine_sintel-20191227-155229/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/X.json
python ./evaluate_for_all_datasets.py /data/dataB/models/Y_PWCNet-I_fine_things-20191230-024005/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/Y.json

#conv33 fused no finetune
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ah/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_ah.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ai/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_ai.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/aw/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_aw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ha/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_ha.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/hi/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_hi.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/hw/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_hw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ia/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_ia.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ih/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_ih.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/iw/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_iw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/wa/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_wa.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/wh/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_wh.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/wi/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/conv33_wi.json

#conv33 fused finetuned chairs
python ./evaluate_for_all_datasets.py /data/dataB/models/0C300_PWCNetConv33Fusion-fine_chairs-20200105-171055/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/0C300.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C301_PWCNetConv33Fusion-fine_chairs-20200105-171258/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/0C301.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C302_PWCNetConv33Fusion-fine_chairs-20200105-171550/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/0C302.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C303_PWCNetConv33Fusion-fine_chairs-20200105-171857/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/0C303.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C306_PWCNetConv33Fusion-fine_chairs-20200105-172316/checkpoint_best.ckpt pwcConv33 /data/dataB/meta/full_evals/0C306.json

# fused no finetune
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ah/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_ah.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ai/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_ai.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/aw/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_aw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ha/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_ha.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hi/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_hi.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hw/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_hw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ia/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_ia.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ih/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_ih.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/iw/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_iw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wa/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_wa.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wh/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_wh.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wi/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/blind_wi.json

# fused and finetuned
python ./evaluate_for_all_datasets.py /data/dataB/models/000_PWCNet-AI_fine_sintel-20191220-133049/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/000.json
python ./evaluate_for_all_datasets.py /data/dataB/models/001_PWCNet-IA_fine_sintel-20191220-134913/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/001.json
python ./evaluate_for_all_datasets.py /data/dataB/models/002_PWCNet-AH_fine_sintel-20191225-203650/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/002.json
python ./evaluate_for_all_datasets.py /data/dataB/models/003_PWCNet-HA_fine_sintel-20191225-203955/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/003.json
python ./evaluate_for_all_datasets.py /data/dataB/models/004_PWCNet-IH_fine_sintel-20191225-204252/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/004.json
python ./evaluate_for_all_datasets.py /data/dataB/models/005_PWCNet-HI_fine_sintel-20191225-204931/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/005.json

# special
python ./evaluate_for_all_datasets.py /data/dataB/models/NF01_PWCNet-H_fine_sintel-20191230-021221/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/NF01.json

# fusion x1Zero no finetune
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ah/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ah.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ai/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ai.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/aw/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_aw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ha/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ha.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hi/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_hi.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hw/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_hw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ia/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ia.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ih/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ih.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/iw/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_iw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wa/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_wa.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wh/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_wh.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wi/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_wi.json

# base pwc models as x1Zero
python ./evaluate_for_all_datasets.py /data/dataB/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_A.json
#x python ./evaluate_for_all_datasets.py /data/dataB/models/B_things_PWCNet-20191122-152857_incomplete/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_B.json
python ./evaluate_for_all_datasets.py /data/dataB/models/F_PWCNet-A_fine_chairs-20191212-133136/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_F.json
python ./evaluate_for_all_datasets.py /data/dataB/models/H_PWCNet-sintel-20191209-150448/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_H.json
python ./evaluate_for_all_datasets.py /data/dataB/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_I.json
python ./evaluate_for_all_datasets.py /data/dataB/models/K_PWCNet-A_fine_things-20191212-133436/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_K.json
python ./evaluate_for_all_datasets.py /data/dataB/models/O_PWCNet-I_fine_kitti-20191226-230605/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_O.json
python ./evaluate_for_all_datasets.py /data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_R.json
python ./evaluate_for_all_datasets.py /data/dataB/models/S_PWCNet-A_fine_KITTI-20191216-125450/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_S.json
python ./evaluate_for_all_datasets.py /data/dataB/models/V_PWCNet-I_fine_chairs-20191230-031321/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_V.json
python ./evaluate_for_all_datasets.py /data/dataB/models/W_PWCNet-kitti-20191216-124247/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_W.json
python ./evaluate_for_all_datasets.py /data/dataB/models/X_PWCNet-I_fine_sintel-20191227-155229/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_X.json
python ./evaluate_for_all_datasets.py /data/dataB/models/Y_PWCNet-I_fine_things-20191230-024005/checkpoint_best.ckpt pwcX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_Y.json

# repeated experiments
python ./evaluate_for_all_datasets.py /data/dataB/models/C_PWCNet-onChairs-20191126-113818/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/C.json
python ./evaluate_for_all_datasets.py /data/dataB/models/H2_PWCNet-sintel-20191227-162133/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/H2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/I2_PWCNet-things-20191230-022450/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/I2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/G_PWCNet-A_fine_sintel-20191212-134449/checkpoint_best.ckpt pwc /data/dataB/meta/full_evals/G.json