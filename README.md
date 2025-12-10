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

## Todo
- Normalisation: The steps are already normalised in the data, i.e. every time step is 0.1 second. But the rest of data like the path moved could still be normalised, its just unclear to me what kind of normalisation would be best.

- Predicting: Since im unsure how to proceed with actual predicting this is still open.
 1. First thing I will try a simple treshold on distance.


## Comments on how to proceed/open questions

The dtw algorithm compares **two** time dependant sequences and return a distance between them. The question is how do we use this to compare different signatures. There are 30 different authors, i currently think we can only do them one by one, otherwise we mess up with normalisation (maybe someone has an idea about that). 

My current ideas:

1. Compute pairwise DTW distances between the 5 genuine signatures. This will give us 25 dtw distances. These can then be used to compute an average distance and standart variation to each other.<br><br>
For each test-signatures test compute all the dtw distances to the 5 genuine signatures, which gives us 5 distances. these need to be combined into a single score, f.e. avarage of the k (2-3) closest distances.<br><br>
We then check if this distance fits into the average distance + standart deviation. While this might work i think we are either going to have tons of false negative or false positives.

2. Work with a knn, this probably doesnt work as we lack samples per writer and since we end up with only one center it would say genuine to all of them. So i discarded this for idea for the moment.

Maybe anyone has a better idea how to proceed with this?

# Implementation choice

# Results

# Impact of normalization