# GryphOne

Hand-written mathematical expression recognition based on discrete diffusion using triplet attention blocks.

## Usage

### Install

Install [PyTorch](https://pytorch.org) 2.4+ and run the following command:

```sh
$ pip install -e .
```

### Models

See [releases](https://github.com/JG1VPP/gryphone/releases).

### Datasets

Download the following datasets:

- [CROHME 2014](https://tc11.cvc.uab.es/datasets/CROHME-2014_2)
- [CROHME 2016](https://tc11.cvc.uab.es/datasets/ICFHR-CROHME-2016_1)
- [CROHME 2019](https://tc11.cvc.uab.es/datasets/ICDAR2019-CROHME-TDF_1)
- [MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting)

### Preprocess

Run [preprocess.py](preprocess.py) as follows:

```sh
$ python3 preprocess.py datasets/crohme_test.yaml
$ python3 preprocess.py datasets/mathwriting.yaml
```

The datasets must be placed in `data` directory as follows:

```sh
$ ls ~/data
crohme/
  CROHME2014_data/
  CROHME2016_data/
  CROHME2019_data/
mathwriting/
  2024/
    test/
      000a4e8ca49c5a1c.inkml
      001083e26028da36.inkml
      0017bb5822bcba69.inkml
      002ae6d5dd4173e4.inkml
      00386113d577085b.inkml
    train/
      00001d1472a8709f.inkml
      00002504391b73b5.inkml
      00003037d3a6d0ba.inkml
      0000fe986018f92a.inkml
      00011d9f03970147.inkml
    valid/
      00000b332dcd6fe5.inkml
      000453d57c3d334d.inkml
      0005d0be8e507b24.inkml
      0005ea8e21185d36.inkml
      00061be0501a1fa8.inkml
pickle/
  gryph_crohme_test.pkl
  gryph_mathwriting.pkl
```

### Training

Run [train.py](train.py) to start training for MathWriting using four GPUs:

```sh
$ torchrun --nproc-per-node=4 train.py configs/mathwriting.py --work-dir ~/work
```

### Inference

Run [infer.py](infer.py) to start inference using the trained model:

```sh
$ python3 infer.py --config ~/work/mathwriting.py --weight ~/work/epoch_60.pth --split test --store results.pkl
```

Use [LgEval](https://univ-nantes.io/crohme/lgeval) for strict evaluation.

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for more details.
