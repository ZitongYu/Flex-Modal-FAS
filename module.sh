module load pytorch/1.9
pip install --user imgaug
pip install --user timm
pip install --user thop

srun -N 1 -n 1 -c 2 -t 36:00:00 --gres=gpu:a100:1 -p gpusmall --account=project_2001654 python3 ~/ICIP_github/train_FAS_ViT_AvgPool_CrossAtten_RGBDIR_P1234.py
 
#srun -N 1 -n 1 -c 2 -t 00:15:00 --gres=gpu:a100:1 -p gputest --account=project_2001654 python3 ~/ICIP_github/train_FAS_ViT_AvgPool_CrossAtten_RGBDIR_P1234.py

