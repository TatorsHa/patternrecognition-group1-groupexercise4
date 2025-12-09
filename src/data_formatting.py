from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import csv


@dataclass
class SignatureSample:
	t: float
	x: float
	y: float
	pressure: float
	penup: int
	azimuth: float
	inclination: float


@dataclass
class SignatureFeatures:
	x: float
	y: float
	vx: float
	vy: float
	pressure: float


def read_signatures(file_path: str | Path) -> List[SignatureSample]:
	path = Path(file_path)
	samples: List[SignatureSample] = []

	with path.open("r", newline="") as f:
		reader = csv.reader(f, delimiter="\t")
		for cols in reader:
			t, x, y, pressure, penup, azimuth, inclination = cols[:7]
			samples.append(
				SignatureSample(
					t=float(t),
					x=float(x),
					y=float(y),
					pressure=float(pressure),
					penup=int(float(penup)),
					azimuth=float(azimuth),
					inclination=float(inclination),
				)
			)

	return samples


def compute_features(samples: List[SignatureSample]) -> List[SignatureFeatures]:
	features: List[SignatureFeatures] = []

	# Use forward differences so velocity ends at 0 for the last sample
	n = len(samples)

	for i in range(n):
		cur = samples[i]
		if i < n - 1:
			nxt = samples[i + 1]
			dt = nxt.t - cur.t
			
            # I dont assume there is a case for this but better be safe.
			if dt <= 0:
				vx = 0.0
				vy = 0.0
			
            #Calculate velocity
			else:
				vx = (nxt.x - cur.x) / dt
				vy = (nxt.y - cur.y) / dt
				
        # Last sample: velocity set to 0 as from that point on the last point the pen stops
		else:
			vx = 0.0
			vy = 0.0

		features.append(
			SignatureFeatures(x=cur.x, y=cur.y, vx=vx, vy=vy, pressure=cur.pressure)
		)

	return features


def load_features_from_tsv(file_path: str | Path) -> List[SignatureFeatures]:
	return compute_features(read_signatures(file_path))
