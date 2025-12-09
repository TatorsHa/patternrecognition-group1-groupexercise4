# patternrecognition-group1-groupexercise4

## Setup of the folders

-------------------------------------
--  data/
--      SignatureVerification/
--          enrollment/     <-- tsv with sample signatures, 5 per author 001-030
--          verification/   <-- tsv with to be tested signatures     
--
--  src/                    <-- folder for all subfunctions
--      data_formatting.py  <-- preparing the data for usage
--      dtw.py              <-- Implementation of the dtw algorithm 
--   
--  main.py                 <-- main executable
-------------------------------------

## todo
- normalisation: the steps are already normalised in the data, i.e. every time step is 0.1 second. but the rest of data like the path moved could still be normalised, its just unclear to me what kind of normalisation would be best.
- predicting: Since im unsure how to proceed with actual predicting this is still open.


## Comments on how to proceed/open questions

The dtw algorithm compares !two! time dependant sequences and return a distance between them. The question is how do we use this to compare different signatures. There are 30 different authors, i currently think we can only do them one by one, otherwise we mess up with normalisation (maybe someone has an idea about that). 

My current ideas:
1. work with a knn, this probably doesnt work as we lack samples per writer and since we end up with only one center it would say genuine to all of them. So i discarded this for idea for the moment.

2.1. Compute pairwise DTW distances between the 5 genuine signatures. this will give us 25 dtw distances. these can then be used to compute an average distance and standart variation to each other.

2.2 For each test-signatures test compute all the dtw distances to the 5 genuine signatures, which gives us 5 distances. these need to be combined into a single score, f.e. avarage of the k (2-3) closest distances.

2.3 We then check if this distance fits into the average distance + standart deviation. While this might work i think we are either going to have tons of false negative or false positives.

Maybe anyone has a better idea how to proceed with this?

