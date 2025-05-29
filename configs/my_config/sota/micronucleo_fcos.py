
_base_ = [
    '../models/fcos.py',
    #'../datasets/voc0712_corrigido_v2_rep1.py',
    '../datasets/micronucleous.py',
    '../default_runtime.py'
]




# model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
model = dict(
    bbox_head=dict(
        num_classes=3
    )
)
#

# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
#max_epochs = 12
max_epochs = 200
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop'
               )
test_cfg = dict(type='TestLoop'
                )


# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=0.1)
]


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

# learning rate
# param_scheduler = [
#     dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]

# # optimizer
# optim_wrapper = dict(
#     optimizer=dict(lr=0.01),
#     paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
#     clip_grad=dict(max_norm=35, norm_type=2))

