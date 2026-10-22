norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 512
out_indices = [5, 6, 7, 8, 9, 10]
model = dict(
    type='SemAlign',
    pretrained='clip_pretrain/ViT-B-16.pt',
    context_length=77,
    backbone=dict(
        type='VPTCLIPVisionTransformer',
        layers=12,
        style='pytorch',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        input_resolution=512,
        out_indices=[5, 6, 7, 8, 9, 10],
        num_tokens=10,
        prompt_dim=768,
        total_d_layer=11),
    text_encoder=dict(
        type='CLIPTextEncoder',
        context_length=77,
        style='pytorch',
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12),
    decode_head=dict(
        type='SemAlignHead',
        img_size=512,
        in_channels=512,
        channels=768,
        num_classes=150,
        num_layers=3,
        num_heads=8,
        use_stages=6,
        embed_dims=512,
        loss_decode=dict(
            type='SegLossPlusRCS',
            num_classes=15,
            dec_layers=3,
            loss_weight=1.0,
            mask_weight=100.0,
            dice_weight=1.0,
            rcs_weight=0.2,
            rcs_temperature=0.1,
            rcs_margin=0.3,
            use_hard_negative=True),
        seen_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        all_idx=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19
        ],
        use_laa=True,
        laa_stages=[2, 3],
        laa_heads=4,
        laa_patch_size=4,
        text_dim=512,
        use_cata=False,
        cata_lite=True,
        use_efm=False,
        use_ccp=True),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)),
    pretrained_text='clip_pretrain/ViT-B-16.pt',
    base_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    novel_class=[15, 16, 17, 18, 19],
    both_class=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ],
    ft_backbone=False,
    exclude_key='prompt',
    load_text_embedding=
    'configs/_base_/datasets/text_embedding/voc12_single.npy')
dataset_type = 'ZeroPascalVOCDataset20'
data_root = 'datasets/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_size=512),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ZeroPascalVOCDataset20',
        data_root='datasets/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='ZeroPascalVOCDataset20',
        data_root='datasets/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, min_size=512),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ZeroPascalVOCDataset20',
        data_root='datasets/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, min_size=512),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=10.0),
            text_encoder=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0),
            ln=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=20001, metric='mIoU')
base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
novel_class = [15, 16, 17, 18, 19]
both_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
]
num_classes = 15
pretrained = 'clip_pretrain/ViT-B-16.pt'
work_dir = './work_dirs/semalign_voc'
