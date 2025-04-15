
_base_ = [
    '../models/atts.py',
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

# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
#max_epochs = 12
max_epochs = 50
# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

train_cfg=dict(
        # assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False)
test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
val_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
