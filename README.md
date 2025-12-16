# patternrecognition-group1-groupexercise4

## Setup of the folders

```
PatternRecog4/
├─ data/
│  └─ SignatureVerification/
│     ├─ enrollment/        # TSVs with 5 genuine signatures per writer (001–030)
│     │  ├─ 001-g-01.tsv
│     │  ├─ 001-g-02.tsv
│     │  └─ ...
│     └─ verification/      # TSVs to be tested (genuine/forged)
│
├─ src/                     # Project modules
│  ├─ data_formatting.py    # Load TSVs and compute features (x,y,vx,vy,pressure)
│  └─ dtw.py                # DTW implementation (Sakoe–Chiba band)
│  └─ treshold_classifier.py# Threshold-based verifier
│
├─ main.py                  # Demo/entry point
├─ pyproject.toml           # Package metadata (editable install)
└─ requirements.txt         # Python dependencies
```

# Implementation choice
We implemented a simple DTW-based approach combined with a threshold classifier. The method compares a candidate sample with genuine samples, takes the minimum distance among the five genuine references, and classifies the sample as a forgery if this distance is greater than a predefined threshold. Otherwise, it is classified as genuine.

In addition, we evaluated the impact of per-signature normalization to assess how much it improves the model’s performance compared to the non-normalized setting.

## Improvement
Performance could be improved by vectorizing parts of the DTW computation or by using optimized numerical libraries. In the current implementation, DTW relies on nested Python loops, which leads to high computational cost when processing many signature comparisons.

The code is not as clean as desired and mainly serves as a prototype. Some computations could be avoided or simplified with a refactoring of the pipeline.

# Results

## No normalisation
![No normalization](./class.png)

## Normalisation
![Normalized](./class_normalized.png)

## Discussion
The results show that normalization has a noticeable effect on the DTW-based threshold classifier. With normalization, the overall accuracy and F1 score are higher than without normalization. Due to the empiric charactere of the threshold one cannot exclude the possibility that the result is just luck.<br><br>
An interesting fact is that the model capture close to 90% of the genuine signature against 68% for the non normalized version, i.e. the algorithm does classify more signatures as genuine compared to the non normalized version. This however comes at the cost of some accuracy, meaning we accept some more forgeries signature as genuine. This problem can be probably balanced with a little bit of finetuning, another classifier or a different normalization.<br><br>
With the tendecies of the two versions known one can specualte about their practical use:<br>

- In non-security-critical applications, where user convenience is more important (such as signing on a tablet for class attendance), the normalized version is more suitable because it rejects fewer genuine signatures. Users are less likely to be frustrated by having their valid signature marked as a forgery.

- In security-critical applications (such as banking, legal contracts, or identity verification), the non-normalized version might be preferable. Even though it rejects more genuine signatures, it accepts fewer forged ones, which reduces the risk of fraud.
