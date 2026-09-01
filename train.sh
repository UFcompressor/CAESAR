
    
    
    
vae_path="./snapshots/fine-tune-eels-all"
train_set="eels,eels2,eels3,eels4"
test_set="eels,eels2,eels3,eels4"

python3 -m pyCAESAR.train_vae3d \
    --save_path=$vae_path \
    --batch_size=32 \
    --iterations=210 \
    --model_dim=16 \
    --lr=0.0004 \
    --beta_start=0.5 \
    --train_set=$train_set \
    --test_set=$test_set \
    --init_beta=0.00001 \
    --end_beta=0.00003 \
    --sr_dim=16 \
    --pretrain="../../CAESAR_C/pretrained/caesar_v.pt"
    
