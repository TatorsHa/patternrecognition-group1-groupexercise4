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
│
├─ main.py                  # Demo/entry point
├─ pyproject.toml           # Package metadata (editable install)
└─ requirements.txt         # Python dependencies
```

# Implementation choice
We implemented a simple DTW-based approach combined with a threshold classifier. The method compares a candidate sample with genuine samples, takes the minimum distance among the five genuine references, and classifies the sample as a forgery if this distance is greater than a predefined threshold, if not it is classified as genuine.<br><br> In a second step, since the threshold classifier achieved acceptable accuracy, we introduced a normalization step to evaluate how much it could further improve the model’s performance.

## Improvement
All the code is in "basic" python, a reimplementation using numpy, can speedup the whole system. 

The code is not as clean as wanted especially because it use class instead of staying the whole time in vectors. A lot of computation can be avoided with a rewriting of the prototype.

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