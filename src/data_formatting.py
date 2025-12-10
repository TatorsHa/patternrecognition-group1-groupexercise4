from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    """Read raw signature samples from a TSV file."""
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


def normalize_samples(samples: List[SignatureSample]) -> List[SignatureSample]:
    if not samples:
        return samples

    # Extract data as numpy arrays
    xy = np.array([[s.x, s.y] for s in samples])
    pressure = np.array([[s.pressure] for s in samples])

    # Normalize x, y with shared scale to preserve aspect ratio
    xy_centered = xy - xy.mean(axis=0)
    scale = xy_centered.std()  # Single scale factor for both dimensions
    if scale == 0:
        scale = 1.0
    xy_normalized = xy_centered / scale

    # Normalize pressure to [0, 1]
    pressure_scaler = MinMaxScaler(feature_range=(0, 1))
    pressure_normalized = pressure_scaler.fit_transform(pressure)

    # Rebuild samples
    normalized: List[SignatureSample] = []
    for i, s in enumerate(samples):
        normalized.append(
            SignatureSample(
                t=s.t,
                x=xy_normalized[i, 0],
                y=xy_normalized[i, 1],
                pressure=pressure_normalized[i, 0],
                penup=s.penup,
                azimuth=s.azimuth,
                inclination=s.inclination,
            )
        )

    return normalized


def compute_features(
    samples: List[SignatureSample], 
    normalize_velocity: bool = True
) -> List[SignatureFeatures]:
    if not samples:
        return []

    n = len(samples)
    
    t = np.array([s.t for s in samples])
    x = np.array([s.x for s in samples])
    y = np.array([s.y for s in samples])
    pressure = np.array([s.pressure for s in samples])

    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)

    dt = np.where(dt <= 0, 1.0, dt)
    
    vx = np.zeros(n)
    vy = np.zeros(n)
    vx[:-1] = dx / dt
    vy[:-1] = dy / dt

    # Normalize velocities
    if normalize_velocity and n > 1:
        velocity_scaler = StandardScaler()
        velocities = np.column_stack([vx, vy])
        velocities_normalized = velocity_scaler.fit_transform(velocities)
        vx = velocities_normalized[:, 0]
        vy = velocities_normalized[:, 1]

    # Build feature list
    features: List[SignatureFeatures] = []
    for i in range(n):
        features.append(
            SignatureFeatures(
                x=x[i],
                y=y[i],
                vx=vx[i],
                vy=vy[i],
                pressure=pressure[i],
            )
        )

    return features


def load_features_from_tsv(
    file_path: str | Path, 
    normalize: bool = True
) -> List[SignatureFeatures]:
    samples = read_signatures(file_path)
    
    if normalize:
        samples = normalize_samples(samples)
    
    return compute_features(samples, normalize_velocity=normalize)


def load_features_from_tsv_raw(file_path: str | Path) -> List[SignatureFeatures]:
    """Load features without normalization (for backward compatibility)."""
    return load_features_from_tsv(file_path, normalize=False)