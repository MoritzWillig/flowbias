export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# base experiments
python ./evaluate_for_all_datasets.py /data/dataB/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/pwc_chairs.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNet-kitti_fixed_aug-20200225-020620/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/pwc_kitti_temp.json
python ./evaluate_for_all_datasets.py /data/dataB/models/pwc_kitti_PWCNet-kitti_fixed_aug-20200225-020620/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/pwc_kitti.json
#x python ./evaluate_for_all_datasets.py /data/dataB/models/B_things_PWCNet-20191122-152857_incomplete/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/B.json
python ./evaluate_for_all_datasets.py /data/dataB/models/D_FlowNet1S-onChairs-20191205-145310/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/D.json
python ./evaluate_for_all_datasets.py /data/dataB/models/E_FlowNet1S-onThings-20191205-115159/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/E.json
python ./evaluate_for_all_datasets.py /data/dataB/models/F_PWCNet-A_fine_chairs-20191212-133136/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/F.json
python ./evaluate_for_all_datasets.py /data/dataB/models/H_PWCNet-sintel-20191209-150448/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/H.json
python ./evaluate_for_all_datasets.py /data/dataB/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/I.json
python ./evaluate_for_all_datasets.py /data/dataB/models/J_FlowNet1S-D_fine_things-20191216-130336/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/J.json
python ./evaluate_for_all_datasets.py /data/dataB/models/K_PWCNet-A_fine_things-20191212-133436/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/K.json
python ./evaluate_for_all_datasets.py /data/dataB/models/L_FlowNet1S-D_fine_chairs-20191218-134755/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/L.json
python ./evaluate_for_all_datasets.py /data/dataB/models/M_FlowNet1S-sintel-20191220-111613/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/M.json
python ./evaluate_for_all_datasets.py /data/dataB/models/N_FlowNet1S-D_fine_sintel-20191218-140407/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/N.json
python ./evaluate_for_all_datasets.py /data/dataB/models/O_PWCNet-I_fine_kitti-20191226-230605/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/O.json
python ./evaluate_for_all_datasets.py /data/dataB/models/P_FlowNet1S-D_fine_kitti-20191227-005059/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/P.json
python ./evaluate_for_all_datasets.py /data/dataB/models/Q_FlowNet1S-E_fine_kitti-20191227-020409/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/Q.json
python ./evaluate_for_all_datasets.py /data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/R.json
python ./evaluate_for_all_datasets.py /data/dataB/models/S_PWCNet-A_fine_KITTI-20191216-125450/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/S.json
python ./evaluate_for_all_datasets.py /data/dataB/models/T_FlowNet1S-sintel-20191227-163340/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/T.json
python ./evaluate_for_all_datasets.py /data/dataB/models/U_FlowNet1S-kitti-20191220-110932/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/U.json
python ./evaluate_for_all_datasets.py /data/dataB/models/V_PWCNet-I_fine_chairs-20191230-031321/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/V.json
python ./evaluate_for_all_datasets.py /data/dataB/models/W_PWCNet-kitti-20191216-124247/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/W.json
python ./evaluate_for_all_datasets.py /data/dataB/models/X_PWCNet-I_fine_sintel-20191227-155229/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/X.json
python ./evaluate_for_all_datasets.py /data/dataB/models/Y_PWCNet-I_fine_things-20191230-024005/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/Y.json
python ./evaluate_for_all_datasets.py /data/dataB/models/Z_FlowNet1S-sintel-20191227-163340/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/Z.json
python ./evaluate_for_all_datasets.py /data/dataB/models/ZZ_FlowNet1S-E_fine_chairs-20200111-180519/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/ZZ.json
python ./evaluate_for_all_datasets.py /data/dataB/models/ZT_FlowNet1S-E_fine_things_res1-20200116-140705/checkpoint_best.ckpt FlowNet1S /data/dataB/meta/full_evals/ZT.json

#conv33 fused no finetune
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ah/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_ah.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ai/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_ai.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/aw/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_aw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ha/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_ha.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/hi/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_hi.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/hw/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_hw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ia/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_ia.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/ih/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_ih.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/iw/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_iw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/wa/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_wa.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/wh/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_wh.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsConv33/wi/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/conv33_wi.json

#conv33 fused finetuned chairs
python ./evaluate_for_all_datasets.py /data/dataB/models/0C300_PWCNetConv33Fusion-fine_chairs-20200105-171055/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C300.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C301_PWCNetConv33Fusion-fine_chairs-20200105-171258/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C301.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C302_PWCNetConv33Fusion-fine_chairs-20200105-171550/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C302.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C303_PWCNetConv33Fusion-fine_chairs-20200105-171857/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C303.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C304_PWCNetConv33Fusion-fine_chairs-20200111-174643/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C304.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C305_res2_PWCNetConv33Fusion-fine_chairs-20200116-141744/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C305.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C306_PWCNetConv33Fusion-fine_chairs-20200105-172316/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C306.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C307_PWCNetConv33Fusion-fine_chairs-20200111-174308/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C307.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C308_PWCNetConv33Fusion-fine_chairs-20200116-020142/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C308.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C309_PWCNetConv33Fusion-fine_chairs-20200116-141109/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C309.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C310_PWCNetConv33Fusion-fine_chairs-20200116-022159/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C310.json
python ./evaluate_for_all_datasets.py /data/dataB/models/0C311_PWCNetConv33Fusion-fine_chairs-20200122-163116/checkpoint_best.ckpt PWCNetConv33Fusion /data/dataB/meta/full_evals/0C311.json

# fused no finetune
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ah/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_ah.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ai/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_ai.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/aw/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_aw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ha/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_ha.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hi/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_hi.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hw/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_hw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ia/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_ia.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ih/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_ih.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/iw/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_iw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wa/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_wa.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wh/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_wh.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wi/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/blind_wi.json

# fused and finetuned
python ./evaluate_for_all_datasets.py /data/dataB/models/000_PWCNet-AI_fine_sintel-20191220-133049/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/000.json
python ./evaluate_for_all_datasets.py /data/dataB/models/001_PWCNet-IA_fine_sintel-20191220-134913/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/001.json
python ./evaluate_for_all_datasets.py /data/dataB/models/002_PWCNet-AH_fine_sintel-20191225-203650/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/002.json
python ./evaluate_for_all_datasets.py /data/dataB/models/003_PWCNet-HA_fine_sintel-20191225-203955/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/003.json
python ./evaluate_for_all_datasets.py /data/dataB/models/004_PWCNet-IH_fine_sintel-20191225-204252/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/004.json
python ./evaluate_for_all_datasets.py /data/dataB/models/005_PWCNet-HI_fine_sintel-20191225-204931/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/005.json

# special
python ./evaluate_for_all_datasets.py /data/dataB/models/NF01_PWCNet-H_fine_sintel-20191230-021221/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/NF01.json

# fusion x1Zero no finetune
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ah/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ah.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ai/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ai.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/aw/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_aw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ha/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ha.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hi/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_hi.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/hw/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_hw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ia/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ia.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/ih/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_ih.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/iw/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_iw.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wa/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_wa.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wh/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_wh.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModels_blind/wi/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_wi.json

# base PWCNet models as x1Zero
python ./evaluate_for_all_datasets.py /data/dataB/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_A.json
#x python ./evaluate_for_all_datasets.py /data/dataB/models/B_things_PWCNet-20191122-152857_incomplete/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_B.json
python ./evaluate_for_all_datasets.py /data/dataB/models/F_PWCNet-A_fine_chairs-20191212-133136/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_F.json
python ./evaluate_for_all_datasets.py /data/dataB/models/H_PWCNet-sintel-20191209-150448/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_H.json
python ./evaluate_for_all_datasets.py /data/dataB/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_I.json
python ./evaluate_for_all_datasets.py /data/dataB/models/K_PWCNet-A_fine_things-20191212-133436/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_K.json
python ./evaluate_for_all_datasets.py /data/dataB/models/O_PWCNet-I_fine_kitti-20191226-230605/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_O.json
python ./evaluate_for_all_datasets.py /data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_R.json
python ./evaluate_for_all_datasets.py /data/dataB/models/S_PWCNet-A_fine_KITTI-20191216-125450/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_S.json
python ./evaluate_for_all_datasets.py /data/dataB/models/V_PWCNet-I_fine_chairs-20191230-031321/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_V.json
python ./evaluate_for_all_datasets.py /data/dataB/models/W_PWCNet-kitti-20191216-124247/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_W.json
python ./evaluate_for_all_datasets.py /data/dataB/models/X_PWCNet-I_fine_sintel-20191227-155229/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_X.json
python ./evaluate_for_all_datasets.py /data/dataB/models/Y_PWCNet-I_fine_things-20191230-024005/checkpoint_best.ckpt PWCNetX1Zero /data/dataB/meta/full_evals/x1ZeroBlind_Y.json

# PWCNet without x1 connection
python ./evaluate_for_all_datasets.py /data/dataB/models/WOX1_chairs_PWCNetWOX1Connection-20200122-164023/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_chairs.json
python ./evaluate_for_all_datasets.py /data/dataB/models/WOX1_PWCNetWOX1Connection-things-20200127-234143/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_things.json
python ./evaluate_for_all_datasets.py /data/dataB/models/WOX1_PWCNetWOX1Connection-sintel-20200127-232828/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_sintel.json
python ./evaluate_for_all_datasets.py /data/dataB/models/WOX1_PWCNetWOX1Connection-kitti-20200128-000101/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti_temp.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_00450.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti_tempA.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_01050.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti_tempB.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_01950.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti_tempC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_iter_03000.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti_tempD.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-kitti_temp_fixed_aug-20200225-090042/checkpoint_latest.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_kitti_tempE.json

# PWCNetWOX1 finetuned
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCNetWOX1Connection-WOX1Chairs_fine_things-20200206-153903/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_chairs_fine_things.json

# pwc without x1 connection between encoder, decoder and context network
python ./evaluate_for_all_datasets.py /data/dataB/models/WOX1Ext_PWCNetWOX1ConnectionExt-onChairs-20200209-181206/checkpoint_best.ckpt PWCNetWOX1ConnectionExt /data/dataB/meta/full_evals/WOX1Ext_chairs.json

# blind fused pwc without X1 connection
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/ct/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_ct.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/cs/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_cs.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/ck/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_ck.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/tc/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_tc.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/ts/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_ts.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/tk/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_tk.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/sc/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_sc.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/st/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_st.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/sk/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_sk.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/kc/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_kc.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/kt/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_kt.json
python ./evaluate_for_all_datasets.py /data/dataB/fusedModelsWOX1Conn_blind/ks/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/WOX1Blind_ks.json

# repeated experiments
python ./evaluate_for_all_datasets.py /data/dataB/models/C_PWCNet-onChairs-20191126-113818/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/C.json
python ./evaluate_for_all_datasets.py /data/dataB/models/H2_PWCNet-sintel-20191227-162133/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/H2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/I2_PWCNet-things-20191230-022450/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/I2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/G_PWCNet-A_fine_sintel-20191212-134449/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/G.json

# expert models WOX1 split CTSK
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert0 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert1 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert2 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_expert2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert3 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_expert3.json

# expert models split failed kitti augmentation
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base02_PWCExpertNet-20200124-000701/checkpoint_best.ckpt CTSKPWCExpertNet02Known /data/dataB/meta/full_evals/expert_split02_failedAug_known.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base02_PWCExpertNet-20200124-000701/checkpoint_best.ckpt CTSKPWCExpertNet02Expert0 /data/dataB/meta/full_evals/expert_split02_failedAug_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base02_PWCExpertNet-20200124-000701/checkpoint_best.ckpt CTSKPWCExpertNet02Expert1 /data/dataB/meta/full_evals/expert_split02_failedAug_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base02_PWCExpertNet-20200124-000701/checkpoint_best.ckpt CTSKPWCExpertNet02Expert2 /data/dataB/meta/full_evals/expert_split02_failedAug_expert2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base02_PWCExpertNet-20200124-000701/checkpoint_best.ckpt CTSKPWCExpertNet02Expert3 /data/dataB/meta/full_evals/expert_split02_failedAug_expert3.json

# expert models WOX1 add CTSK
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddNoExpert /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_no_expert.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert0 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert1 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert2 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_expert2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert3 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_expert3.json

# expert models WOX1 linAdd CTSK
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307/checkpoint_best.ckpt CTKSPWCExpertLinAddNet01WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_linAdd01_known.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307/checkpoint_best.ckpt CTKSPWCExpertLinAddNet01WOX1NoExpert /data/dataB/meta/full_evals/expertWOX1_CTSK_linAdd01_no_expert.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307/checkpoint_best.ckpt CTKSPWCExpertLinAddNet01WOX1Expert0 /data/dataB/meta/full_evals/expertWOX1_CTSK_linAdd01_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307/checkpoint_best.ckpt CTKSPWCExpertLinAddNet01WOX1Expert1 /data/dataB/meta/full_evals/expertWOX1_CTSK_linAdd01_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307/checkpoint_best.ckpt CTKSPWCExpertLinAddNet01WOX1Expert2 /data/dataB/meta/full_evals/expertWOX1_CTSK_linAdd01_expert2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_linAdd01_CTKS_PWCExpertLinAddNetWOX1-20200321-035307/checkpoint_best.ckpt CTKSPWCExpertLinAddNet01WOX1Expert3 /data/dataB/meta/full_evals/expertWOX1_CTSK_linAdd01_expert3.json

# expert models WOX1 add CTS
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331/checkpoint_best.ckpt CTSPWCExpertNet01WOX1AddNoExpert /data/dataB/meta/full_evals/expertWOX1_CTS_add01_no_expert.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331/checkpoint_best.ckpt CTSPWCExpertNet01WOX1AddExpert0 /data/dataB/meta/full_evals/expertWOX1_CTS_add01_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331/checkpoint_best.ckpt CTSPWCExpertNet01WOX1AddExpert1 /data/dataB/meta/full_evals/expertWOX1_CTS_add01_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNetWOX1-20200315-002331/checkpoint_best.ckpt CTSPWCExpertNet01WOX1AddExpert2 /data/dataB/meta/full_evals/expertWOX1_CTS_add01_expert2.json

# expert models WOX1 add encoder only CTSK
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437/checkpoint_best.ckpt CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyKnown /data/dataB/meta/full_evals/expertWOX1_encoder_only_CTSK_add01_known.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437/checkpoint_best.ckpt CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyNoExpert /data/dataB/meta/full_evals/expertWOX1_encoder_only_CTSK_add01_no_expert.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437/checkpoint_best.ckpt CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert0 /data/dataB/meta/full_evals/expertWOX1_encoder_only_CTSK_add01_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437/checkpoint_best.ckpt CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert1 /data/dataB/meta/full_evals/expertWOX1_encoder_only_CTSK_add01_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437/checkpoint_best.ckpt CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert2 /data/dataB/meta/full_evals/expertWOX1_encoder_only_CTSK_add01_expert2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_encoder_only_add01_CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly-20200326-132437/checkpoint_best.ckpt CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert3 /data/dataB/meta/full_evals/expertWOX1_encoder_only_CTSK_add01_expert3.json

# expert models add failed kitti augmentation
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNet-20200124-174956/checkpoint_best.ckpt CTSKPWCExpertNet01AddKnown /data/dataB/meta/full_evals/expert_add01_failedAug_known.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNet-20200124-174956/checkpoint_best.ckpt CTSKPWCExpertNet01AddNoExpert /data/dataB/meta/full_evals/expert_add01_failedAug_no_expert.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNet-20200124-174956/checkpoint_best.ckpt CTSKPWCExpertNet01AddExpert0 /data/dataB/meta/full_evals/expert_add01_failedAug_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNet-20200124-174956/checkpoint_best.ckpt CTSKPWCExpertNet01AddExpert1 /data/dataB/meta/full_evals/expert_add01_failedAug_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNet-20200124-174956/checkpoint_best.ckpt CTSKPWCExpertNet01AddExpert2 /data/dataB/meta/full_evals/expert_add01_failedAug_expert2.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNet-20200124-174956/checkpoint_best.ckpt CTSKPWCExpertNet01AddExpert3 /data/dataB/meta/full_evals/expert_add01_failedAug_expert3.json

#expert model split WOX1 best trace
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_01.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter01.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_10.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter10.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_20.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter20.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_30.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter30.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_40.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter40.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_50.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter50.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_52.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter52.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_54.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter54.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_56.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter56.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_58.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter58.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_iter_60.ckpt CTSKPWCExpertNet02WOX1Known /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_known_iter60.json

#expert model add WOX1 best trace
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_01.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter01.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_10.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter10.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_20.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter20.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_30.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter30.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_40.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter40.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_50.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter50.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_52.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter52.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_54.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter54.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_56.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter56.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_58.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter58.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_iter_60.ckpt CTSKPWCExpertNet01WOX1AddKnown /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_known_iter60.json

#CTSK WOX1 trace
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_01.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter01.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_10.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter10.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_20.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter20.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_30.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter30.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_40.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter40.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_50.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter50.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_52.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter52.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_54.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter54.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_56.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter56.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_58.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter58.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_iter_60.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK_iter60.json

# expert models add CTS
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddNoExpert /data/dataB/meta/full_evals/expert_CTS_add01_no_expert.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert0 /data/dataB/meta/full_evals/expert_CTS_add01_expert0.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert1 /data/dataB/meta/full_evals/expert_CTS_add01_expert1.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert2 /data/dataB/meta/full_evals/expert_CTS_add01_expert2.json

# pwc trained on CTSK
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_noExpert_failedaugment_PWCNet-20200127-234847/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/pwc_on_CTSK_failedaugment.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_noExpert_PWCNet-20200225-093211/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/pwc_on_CTSK.json

# pwcWOX1 trained on multiple datasets
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTSK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/pwcWOX1_mixed_batch_CTSK_PWCNetWOX1Connection-20200407-220817/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_mixed_batch_on_CTSK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/pwc_noExperts_CTS_wox1_PWCNetWOX1Connection-20200311-213609/checkpoint_best.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_on_CTS.json

# iteration inspection
python ./evaluate_for_all_datasets.py /data/dataB/models/iter_PWCNet-chairs_148/checkpoint_iter_148.ckpt PWCNet /data/dataB/meta/full_evals/pwc_chairs_iter_148.json

# pwc down sampling encoder
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_DSEncoder_PWCNetDSEncoder-onChairs-20200209-190403/checkpoint_best.ckpt PWCNetDSEncoder /data/dataB/meta/full_evals/pwcDSEncoder_chairs.json

# pwc residual flow
python ./evaluate_for_all_datasets.py /data/dataB/models/pwc_residual_flow_CTSK_PWCNetResidualFlow-20200326-131601/checkpoint_best.ckpt PWCNetResidualFlow /data/dataB/meta/full_evals/pwc_residual_flow_CTSK.json

# pwc secondary flow
python ./evaluate_for_all_datasets.py /data/dataB/models/pwcWOX1_CTSK_secondary_flow_PWCNetWOX1SecondaryFlow-20200330-082846/checkpoint_best.ckpt PWCNetWOX1SecondaryFlow /data/dataB/meta/full_evals/pwcWOX1_secondary_flow_CTSK.json

# pwc delta target
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_delta_PWCNet-onChairs-20200220-122429/checkpoint_best.ckpt PWCNet /data/dataB/meta/full_evals/pwcDelta_chairs.json

# pwc chairs every eval
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_001.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_001.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_011.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_011.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_021.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_021.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_031.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_031.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_041.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_041.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_051.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_051.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_061.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_061.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_071.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_071.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_081.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_081.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_091.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_091.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_101.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_101.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_111.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_111.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_121.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_121.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_131.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_131.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_141.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_141.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_151.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_151.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_161.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_161.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_171.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_171.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_181.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_181.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_191.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_191.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_201.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_201.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWC_EVERY_PWCNet-onChairs-20200203-022421/checkpoint_iter_209.ckpt PWCNet /data/dataB/meta/full_evals/pwc_every_chairs_209.json

# pwc WOX1 chairs every eval
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_001.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_001.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_011.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_011.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_021.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_021.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_031.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_031.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_041.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_041.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_051.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_051.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_061.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_061.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_071.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_071.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_081.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_081.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_091.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_091.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_101.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_101.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_111.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_111.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_121.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_121.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_131.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_131.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_141.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_141.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_151.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_151.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_161.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_161.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_171.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_171.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_181.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_181.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_191.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_191.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_201.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_201.json
python ./evaluate_for_all_datasets.py /data/dataB/models/PWCWOX1_EVERY_PWCNetWOX1Connection-onChairs-20200203-022540/checkpoint_iter_209.ckpt PWCNetWOX1Connection /data/dataB/meta/full_evals/pwcWOX1_every_chairs_209.json


#pwc CTS expert model fused eval
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert00 /data/dataB/meta/full_evals/expert_CTS_add01_CC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert01 /data/dataB/meta/full_evals/expert_CTS_add01_CT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert02 /data/dataB/meta/full_evals/expert_CTS_add01_CS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert10 /data/dataB/meta/full_evals/expert_CTS_add01_TC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert11 /data/dataB/meta/full_evals/expert_CTS_add01_TT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert12 /data/dataB/meta/full_evals/expert_CTS_add01_TS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert20 /data/dataB/meta/full_evals/expert_CTS_add01_SC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert21 /data/dataB/meta/full_evals/expert_CTS_add01_ST.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_CTS_PWCExpertAddNet-20200210-223344/checkpoint_best.ckpt CTSPWCExpertNet01AddExpert22 /data/dataB/meta/full_evals/expert_CTS_add01_SS.json


#CTKS WOX1 fused expert split models
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert00 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_CC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert01 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_CT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert02 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_CS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert03 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_CK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert10 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_TC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert11 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_TT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert12 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_TS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert13 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_TK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert20 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_SC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert21 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_ST.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert22 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_SS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert23 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_SK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert30 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_KC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert31 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_KT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert32 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_KS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_base_PWCExpertNetWOX1-20200227-012926/checkpoint_best.ckpt CTSKPWCExpertNet02WOX1Expert33 /data/dataB/meta/full_evals/expertWOX1_CTSK_split02_KK.json

#CTKS WOX1 fused expert split models
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert00 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_CC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert01 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_CT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert02 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_CS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert03 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_CK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert10 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_TC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert11 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_TT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert12 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_TS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert13 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_TK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert20 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_SC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert21 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_ST.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert22 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_SS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert23 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_SK.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert30 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_KC.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert31 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_KT.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert32 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_KS.json
python ./evaluate_for_all_datasets.py /data/dataB/models/expert_add01_PWCExpertAddNetWOX1-20200227-013322/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert33 /data/dataB/meta/full_evals/expertWOX1_CTSK_add01_KK.json

python ./evaluate_for_all_datasets.py /data/dataB/unifiedModelsExpert/unified_avg_expertWOX1_CTSK_add01_expert/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert0 /data/dataB/meta/full_evals/unified_avg_expertWOX1_CTSK_add01_expert.json
python ./evaluate_for_all_datasets.py /data/dataB/unifiedModelsExpert/unifiedCTS_avg_expertWOX1_CTSK_add01_expert/checkpoint_best.ckpt CTSKPWCExpertNet01WOX1AddExpert0 /data/dataB/meta/full_evals/unifiedCTS_avg_expertWOX1_CTSK_add01_expert.json


