custom_imports = dict(
    imports=[
        "gryph.data",
        "gryph.loss",
        "gryph.model",
        "gryph.score",
        "gryph.vocab",
    ]
)

len_tex = 150

model = dict(
    type="FormulaScanner",
    batcher=dict(
        type="FormulaBatcher",
        lexicon=dict(
            type="FormulaVocab",
            load="alphabet/mathwriting.txt",
            skip="<SKIP>",
            mask="<MASK>",
        ),
    ),
    encoder=dict(
        type="FormulaEncoder",
        backbone="facebook/dino-vits8",
        add_pooling_layer=False,
    ),
    decoder=dict(
        type="FormulaRefiner",
        model=dict(
            type="FormulaDecoder",
            blocks=[
                dict(
                    att1=dict(type="GlobalAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
                dict(
                    att1=dict(type="GlobalAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
                dict(
                    att1=dict(type="GlobalAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
                dict(
                    att1=dict(type="GlobalAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
                dict(
                    att1=dict(type="GlobalAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
            ],
            heads=8,
            d_model=384,
            dropout=0.3,
            updater=dict(type="FormulaUpdater"),
        ),
        steps=50,
    ),
    handler=dict(
        type="FormulaHandler",
        len_tex=len_tex,
        outputs=["sym", "mod"],
        targets=["sym", "mod", "name"],
    ),
    losses=[
        # sym
        dict(type="CELoss", key="sym1"),
        dict(type="CELoss", key="sym2"),
        dict(type="KLLoss", key="sym1", mut="sym2"),
        dict(type="KLLoss", key="sym2", mut="sym1"),
        # mod
        dict(type="CELoss", key="mod1"),
        dict(type="CELoss", key="mod2"),
        dict(type="KLLoss", key="mod1", mut="mod2"),
        dict(type="KLLoss", key="mod2", mut="mod1"),
    ],
)

pipeline = [
    dict(type="ScaleInk", w=224, h=224),
    dict(type="PaintInk", w=224, h=224, fill=(1, 1, 1), line=1),
    dict(type="LabelTeX", split=True),
    dict(
        type="Annotate",
        keys=["img"],
        meta=["sym", "mod", "name"],
    ),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=0,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="FormulaDataset",
        ann_file="~/data/pickle/gryph_mathwriting.pkl",
        filter_cfg=dict(split="train"),
        pipeline=pipeline,
        test_mode=False,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=0,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="FormulaDataset",
        ann_file="~/data/pickle/gryph_mathwriting.pkl",
        filter_cfg=dict(split="valid"),
        pipeline=pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=0,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="FormulaDataset",
        ann_file="~/data/pickle/gryph_mathwriting.pkl",
        filter_cfg=dict(split="test"),
        pipeline=pipeline,
        test_mode=True,
    ),
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=60,
)

val_cfg = dict(type="ValLoop")

test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.001,
    ),
    clip_grad=dict(
        max_norm=10,
        norm_type=2,
    ),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.2,
        begin=0,
        end=100,
        by_epoch=False,
    ),
    dict(
        type="MultiStepLR",
        milestones=[50],
        gamma=0.1,
        by_epoch=True,
    ),
]

val_evaluator = dict(type="CER", prefix="tex")

test_evaluator = dict(type="CER", prefix="tex")

launcher = "pytorch"
