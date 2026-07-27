# train/serve skew test

- n: 16 | seed: 42
- pt: `model/weights/best.pt`
- coreml: `benchmarks/models/best_fp16.mlpackage`
- conf: 0.25 | imgsz: 640

## per-image
| image | pt_class | coreml_class | top1 agree |
|---|---|---|---|
| IMG_0813_JPG.rf.507e984677d50ec81055e456cd225cd2.jpg | 1 | 1 | True |
| IMG_0596_JPG.rf.84275dfdc4f3b43d9012bc967ceeec26.jpg | 0 | 0 | True |
| IMG_0572_JPG.rf.18b1a791db05209239351a5210d0bb54.jpg | 0 | 0 | True |
| IMG_0833_JPG.rf.94826a7020a29618dbbc1417a6369c5a.jpg | 2 | 2 | True |
| IMG_0674_JPG.rf.28b5fc92a5f324d859764149508fdf44.jpg | 0 | 0 | True |
| IMG_0666_JPG.rf.38af46c65bd8119a7c4aed149cdef94f.jpg | 0 | 0 | True |
| IMG_0660_JPG.rf.c692140e59c70890261ab75b9031adb9.jpg | 0 | 0 | True |
| IMG_0597_JPG.rf.bc58e4be8d64493719ea259a7ef7def3.jpg | 0 | 0 | True |
| IMG_0864_JPG.rf.e08a2c87cdef1d65c440b27d86ab44b8.jpg | 2 | 2 | True |
| IMG_0595_JPG.rf.7ba36afd1a46f44a09eb03e20bd9addb.jpg | 0 | 0 | True |
| IMG_0818_JPG.rf.103ee0dc61a2c735efc0d7cdec54d03e.jpg | 1 | 1 | True |
| IMG_0772_JPG.rf.27a3a13f4fb74d45e9a4335046a3625b.jpg | 1 | 1 | True |
| IMG_0585_JPG.rf.f298093cb7b266a83d17f2794b40cdc3.jpg | 0 | 0 | True |
| IMG_0797_JPG.rf.13a0e20acf179bb9c55344b439d66713.jpg | 1 | 1 | True |
| IMG_0738_JPG.rf.2fa54076dda0efbc4417f275f85816c7.jpg | 3 | 3 | True |
| IMG_0579_JPG.rf.ff9d78540f2cce251a1bda3a99794b63.jpg | 0 | 0 | True |

- top-1 agreement: 16/16 = 1.000 (threshold 0.95): PASS
- mean|Δconf| on matched top-1 (IoU≥0.5): 0.0049 over 16 matched boxes (threshold 0.10): PASS
- verdict: PASS

