from pathlib import Path
from data_formatting import load_features_from_tsv
from dtw import dtw_distance, to_feature_vectors

def main():
	root = Path(__file__).resolve().parent
	enrollment = root / "data" / "SignatureVerification" / "enrollment"

	a = load_features_from_tsv(enrollment / "001-g-01.tsv")
	b = load_features_from_tsv(enrollment / "001-g-02.tsv")
	dist = dtw_distance(to_feature_vectors(a), to_feature_vectors(b), window=3)
	print(f"DTW distance (window=3) between 001-g-01 and 001-g-02: {dist:.3f}")

if __name__ == "__main__":
	main()
