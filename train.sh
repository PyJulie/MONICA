python main.py --config ./configs/medmnist/isic_ERM.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0
python main.py --config ./configs/medmnist/isic_cRT_RS.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0  --model.resume_path ./outputs/organsmnist/10_ERM_224_resnet50_True_256_1_50_strong/best.pt
python main.py --config ./configs/medmnist/isic_GCL_1st.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0
python main.py --config ./configs/medmnist/isic_GCL_2nd.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0  --model.resume_path ./outputs/organsmnist/10_GCL_224_resnet50_True_256_1_50_strong/best.pt
python main.py --config ./configs/medmnist/isic_DisAlign.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0  --model.resume_path ./outputs/organsmnist/10_ERM_224_resnet50_True_256_1_50_strong/best.pt
python main.py --config ./configs/medmnist/isic_KNN.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0  --model.resume_path ./outputs/organsmnist/10_ERM_224_resnet50_True_256_1_50_strong/best.pt
python main.py --config ./configs/medmnist/isic_LWS.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0  --model.resume_path ./outputs/organsmnist/10_ERM_224_resnet50_True_256_1_50_strong/best.pt
python main.py --config ./configs/medmnist/isic_MixUp.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0
python main.py --config ./configs/medmnist/isic_MiSLAS.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 0  --model.resume_path ./outputs/organsmnist/10_MixUp_224_resnet50_True_256_1_50_strong/best.pt
python main.py --config ./configs/medmnist/isic_RS.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_De-Confound.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_SEQLLoss.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_Logits_Adjust_Loss.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_BalancedSoftmax.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_VSLoss.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_CBLoss.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_RangeLoss.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_Focal.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_SAM.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 1 
python main.py --config ./configs/medmnist/isic_WeightedSoftmax.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_SADE.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_RSG.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_LADELoss.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_LDAM.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_Logits_Adjust_Posthoc.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_CBFocal.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_PriorCELoss.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_T-Norm.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_RW.yml ./configs/medmnist/base_organsmnist.yml --cuda.gpu_id 2 
python main.py --config ./configs/medmnist/isic_BBN.yml ./configs/medmnist/base_organcmnist.yml --cuda.gpu_id 3 
