# set to wherever pix2pixHD root is
pix2pixHD_root="/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD"
decay_epochs=10        # 10 epochs for 1 cycle from start_lr to 0
start_lr=0.001
start_epoch=100        # resume training here
update_lr=100          # update LR every 100 iterations (fractional LR update)


train_command = "
    python3 train.py \
        --dataroot=$pix2pixHD_root/datasets/cityscapes/             \
        --checkpoints_dir=$pix2pixHD_root/checkpoints/              \
        --name label2city_512p_feat                                 \
        --instance_feat                                             \
        --continue_train                                            \
        --started_epoch=$start_epoch                                \
        --lr=$start_lr                                              \
        --niter=$start_epoch                                        \
        --niter_decay=$decay_epochs                                 \
        --cos_decay                                                 \
        --cos_decay_update_iters=$update_lr                         \
"

# added options:
# started_epoch (set to constant, on restart can correctly update LR from last place)
# cos_decay (bool flag to use cosine decay)
# cos_decay_udpate_iters (number of iterations (sub-epoch) between updates)