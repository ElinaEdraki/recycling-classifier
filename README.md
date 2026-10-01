# Recycling Material Classifier (MobileNetV2)

An image classifier that identifies the material of a photographed waste item — **cardboard, glass, metal, paper or plastic** — to help reduce contamination in recycling bins. Built with PyTorch using a pretrained MobileNetV2, with a focus on data augmentation and on how well the model generalises to real-world photos.

Originally completed for SIT319 (Deakin University).

## Dataset

[Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) on Kaggle (based on TrashNet), downloaded automatically via `kagglehub`. The `trash` class was dropped, leaving ~2,390 images across 5 classes, split 70/15/15 (stratified).

## Approach

1. **Baseline** – MobileNetV2 (ImageNet weights) with a new 5-class head. Adam (lr 1e-4), batch size 32, 10 epochs, best checkpoint chosen by validation macro-F1.
2. **Augmentation** – Training-only augmentation pipeline, compared against the baseline at equal training cost, plus a profile of where pipeline time goes.
3. **Domain shift** – Failure analysis, then evaluation on a new-domain test set of my own phone photos (real kitchen backgrounds, uncontrolled lighting, varied angles), followed by a targeted stronger-augmentation fix.

## Results

| Model | Test accuracy | Test macro-F1 | New-domain accuracy | New-domain macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.936 | 0.936 | – | – |
| + Augmentation | 0.939 | 0.939 | 0.400 | 0.357 |
| + Targeted improvement | 0.925 | 0.924 | **0.629** | **0.614** |

The main takeaway: near-94% accuracy on the studio-style dataset dropped to 40% on real photos. Targeted augmentation recovered a large part of that gap (+23 points) at the cost of about 1 point on the original test set.

The notebook also includes a separate section reproducing and investigating results from an ML-security paper using GPT-2.

## Running it

The notebook was written for **Google Colab** with a GPU runtime.

1. Open `recycling_classifier.ipynb` in Colab.
2. Run the cells in order. The first cells mount Google Drive and set `PROJECT_DIR`; change that path if you want to store things elsewhere.
3. The dataset downloads automatically. The new-domain photos are not included in this repo, so the Task 3.2 cells need your own images in the expected folder.

To run locally instead, install the requirements and remove the `google.colab` drive-mount cell:

```bash
pip install -r requirements.txt
```
