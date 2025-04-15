
# dataset settings
dataset_type = 'CocoDataset'
classes = ('BN', 'BNMN', 'MN')

data_root = '/mnt/hd_pesquisa/pesquisa/datasets/micronucleo_kaggle/'

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
    # dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(
    type='RandomChoiceResize',
    scales=[(800, 600), (1000, 600), (1333, 800)],
    keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), allow_negative_crop=True),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
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
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    pin_memory=True,   
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
                    # ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    ann_file = 'annotations/train.json',
                    #ann_file='VOC2007/ImageSets/Main/trainval_debug_nano.txt',  # head -n 10 trainval.txt_ > trainval_debug_nano.txt 
                    # data_prefix=dict(sub_data_root='annotations/'),
                    data_prefix=dict(img=data_root + 'images/train/'),
                    metainfo=dict(classes=('BN', 'BNMN', 'MN')),
                    #img_prefix=(data_root + 'images/train/'),
                    # data_prefix=dict(sub_data_root='images/train/'),
                    serialize_data=False,
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
                    pipeline=train_pipeline,
                    backend_args=backend_args)
                # dict(
                #     type=dataset_type,
                #     data_root=data_root,
                #     ann_file='VOC2012/ImageSets/Main/trainval.txt',
                #     data_prefix=dict(sub_data_root='VOC2012/'),
                #     serialize_data=True,  
                #     filter_cfg=dict(
                #         filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
                #     pipeline=train_pipeline,
                #     backend_args=backend_args)
            ])
            ))


val_dataloader = dict(
    batch_size=2,
    # classes=classes,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file='VOC2007/ImageSets/Main/test.txt',
        ann_file = 'annotations/val.json',
        # data_prefix=dict(sub_data_root='annotations/'),
        data_prefix=dict(img=data_root + 'images/val/'),
        # data_prefix=dict(data_root + 'images/val/'),
        # data_prefix=dict(sub_data_root='images/val/'),
        # img_prefix=(data_root + 'images/val/'),
        metainfo=dict(classes=('BN', 'BNMN', 'MN')),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=2,
    # classes=classes,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file='VOC2007/ImageSets/Main/test.txt',
        ann_file = 'annotations/test.json',
        # data_prefix=dict(sub_data_root='annotations/'),
        data_prefix=dict(img=data_root + 'images/test/'),
        #img_prefix=(data_root + 'images/test/'),
        # data_prefix=dict(data_root + 'images/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('BN', 'BNMN', 'MN')),
        backend_args=backend_args))

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
# val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator 

