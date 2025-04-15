# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically Infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/segmentation/VOCdevkit/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#         'data/': 's3://openmmlab/datasets/segmentation/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5), #debug
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(

    batch_size=2,
    # batch_size=8,
    num_workers=2,
    # persistent_workers=True,
    persistent_workers=False,   # FILIPE DEBUG
    sampler=dict(type='DefaultSampler', shuffle=True),
    # sampler=dict(type='DefaultSampler', shuffle=False), #FILIPE DEBUG
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    pin_memory=True,   
    #pin_memory=False,    # FILIPE DEBUGGING
    dataset=dict(
        type='RepeatDataset',
        #times=3,
        times=1,
        dataset=dict(
            type='ConcatDataset',
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    #ann_file='VOC2007/ImageSets/Main/trainval_debug_nano.txt',  # head -n 10 trainval.txt_ > trainval_debug_nano.txt 
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    serialize_data=False,  # Define como False (Filipe)
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
                    pipeline=train_pipeline,
                    backend_args=backend_args),
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    serialize_data=False,  # Define como  [FILIPE]
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
                    pipeline=train_pipeline,
                    backend_args=backend_args)
            ])))


# debug_train_dataloader = dict(

#     batch_size=2,
#     # batch_size=8,
#     num_workers=2,
#     persistent_workers=True,
#     # persistent_workers=False,   # FILIPE DEBUG
#     #sampler=dict(type='DefaultSampler', shuffle=True),
#     sampler=dict(type='DefaultSampler', shuffle=False), #FILIPE DEBUG
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     pin_memory=True,   
#     #pin_memory=False,    # FILIPE DEBUGGING
#     dataset=dict(
#         type='RepeatDataset',
#         #times=3,
#         times=1,
#         dataset=dict(
#             type='ConcatDataset',
#             # VOCDataset will add different `dataset_type` in dataset.metainfo,
#             # which will get error if using ConcatDataset. Adding
#             # `ignore_keys` can avoid this error.
#             ignore_keys=['dataset_type'],
#             datasets=[
#                 dict(
#                     type=dataset_type,
#                     data_root=data_root,
#                     ann_file='VOC2007/ImageSets/Main/trainval.txt',
#                     #ann_file='VOC2007/ImageSets/Main/trainval_debug_nano.txt',  # head -n 10 trainval.txt_ > trainval_debug_nano.txt 
#                     data_prefix=dict(sub_data_root='VOC2007/'),
#                     # test_mode=True,
#                     # serialize_data=False,  # Define como False (Filipe)
#                     filter_cfg=dict(
#                         filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
#                     pipeline=train_pipeline,
#                     backend_args=backend_args),
#                 dict(
#                     type=dataset_type,
#                     data_root=data_root,
#                     ann_file='VOC2012/ImageSets/Main/trainval.txt',
#                     data_prefix=dict(sub_data_root='VOC2012/'),
#                     # test_mode=True,
#                     # serialize_data=False,  # Define como  [FILIPE]
#                     filter_cfg=dict(
#                         filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
#                     pipeline=train_pipeline,
#                     backend_args=backend_args)
#             ])))

# debug_train_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='VOC2007/ImageSets/Main/trainval.txt',
#         data_prefix=dict(sub_data_root='VOC2007/'),
#         test_mode=True,
#         #pipeline=test_pipeline,
#         pipeline=train_pipeline,
#         backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
# test_dataloader = debug_train_dataloader #debug FILIPE
# val_dataloader = debug_train_dataloader  # debug FILIPE

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator 

