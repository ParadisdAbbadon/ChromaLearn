import os
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import datalad.api as dl
import joblib
from aeon.transformations.collection.convolution_based import MiniRocket
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASETS = [
    {"url": "https://github.com/OpenNeuroDatasets/ds007118.git", "path": "./ds007118", "id": "ds007118"},
    {"url": "https://github.com/OpenNeuroDatasets/ds007119.git", "path": "./ds007119", "id": "ds007119"},
    {"url": "https://github.com/OpenNeuroDatasets/ds007120.git", "path": "./ds007120", "id": "ds007120"},
]
ROCKET_PATH = "./minirocket.joblib"
COLLECTION  = "ieeg_patterns"
EPOCH_TMIN  = -0.5   # seconds before event onset
EPOCH_TMAX  =  3.0   # seconds after event onset
VECTOR_SIZE =  9996  # MiniRocket default output dim
EVENT_ID    = {"seizure": 1, "interictal": 2}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def read_env(path=os.path.join(os.path.dirname(__file__), "..", ".env")) -> dict:
    """Parse a .env file and return a dict of key/value pairs."""
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    missing = {"QDRANT_URL", "QDRANT_API_KEY"} - env.keys()
    if missing:
        raise KeyError(f".env is missing required keys: {missing}")
    return env


def clone_dataset(url: str, path: str) -> dl.Dataset:
    """Clone a DataLad dataset (metadata only) if not already present."""
    if not os.path.exists(path):
        ds = dl.clone(source=url, path=path)
        print(f"Cloned {path} (metadata only)")
    else:
        ds = dl.Dataset(path)
        print(f"Dataset {path} already cloned")
    return ds


def get_epochs_for_subject(
    subject: str,
    dataset: dl.Dataset,
    root: str,
    dataset_id: str,
) -> tuple[np.ndarray | None, list | None]:
    """
    Lazy-fetch one subject's iEEG files, extract seizure/interictal epochs,
    then drop the local files to free disk space.

    Returns (epochs_array, metadata_list) where epochs_array has shape
    (n_epochs, n_channels, n_timepoints), or (None, None) if no usable data.
    """
    bids_path   = BIDSPath(subject=subject, datatype="ieeg", root=root)
    subject_dir = os.path.abspath(os.path.join(root, f"sub-{subject}"))
    dataset.get(subject_dir)

    try:
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
    except Exception as e:
        print(f"  Could not read sub-{subject}: {e}")
        dataset.drop(subject_dir)
        return None, None

    events_files = [f for f in os.listdir(subject_dir) if f.endswith("_events.tsv")]
    if not events_files:
        print(f"  No events.tsv for sub-{subject}, skipping")
        dataset.drop(subject_dir)
        return None, None

    all_epochs_data = []
    all_metadata    = []

    for ef in events_files:
        events_df = pd.read_csv(os.path.join(subject_dir, ef), sep="\t")
        events_df = events_df[
            events_df["trial_type"].str.lower().isin(EVENT_ID.keys())
        ]
        if events_df.empty:
            continue

        sfreq = raw.info["sfreq"]
        mne_events = np.column_stack([
            (events_df["onset"] * sfreq).astype(int),
            np.zeros(len(events_df), dtype=int),
            events_df["trial_type"].str.lower().map(EVENT_ID).fillna(0).astype(int),
        ])
        mne_events = mne_events[mne_events[:, 2] != 0]

        epochs = mne.Epochs(
            raw, mne_events, event_id=EVENT_ID,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=None, preload=True, verbose=False,
        )
        all_epochs_data.append(epochs.get_data())  # (n, ch, t)

        for row in events_df.itertuples():
            all_metadata.append({
                "dataset_id":  dataset_id,
                "subject":     subject,
                "event_type":  row.trial_type.lower(),
                "onset_sec":   float(row.onset),
                "source_file": ef,
            })

    dataset.drop(subject_dir)

    if not all_epochs_data:
        return None, None

    return np.concatenate(all_epochs_data, axis=0), all_metadata


def embed(
    epochs_array: np.ndarray,
    rocket: MiniRocket | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, MiniRocket]:
    """
    Embed a (n_epochs, n_channels, n_timepoints) array using MiniRocket.
    Pass fit=True (or rocket=None) to fit a new transformer.
    Returns L2-normalised features and the (possibly new) rocket instance.
    """
    if fit or rocket is None:
        rocket = MiniRocket(random_state=42)
        rocket.fit(epochs_array)
    features = rocket.transform(epochs_array)
    norms    = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norms == 0, 1, norms)
    return features, rocket


def query_similar(
    query_epochs_array: np.ndarray,
    qdrant: QdrantClient,
    n_results: int = 5,
) -> list:
    """
    Find the most similar stored patterns to a given epoch array.
    query_epochs_array shape: (n_epochs, n_channels, n_timepoints)
    """
    rocket = joblib.load(ROCKET_PATH)
    features, _ = embed(query_epochs_array, rocket=rocket)

    results = []
    for vec in features:
        hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=vec.tolist(),
            limit=n_results,
        )
        results.append(hits)
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    env = read_env()
    qdrant = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"])

    if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION}")

    rocket   = None
    point_id = 0

    for ds_info in DATASETS:
        dataset  = clone_dataset(ds_info["url"], ds_info["path"])
        subjects = get_entity_vals(ds_info["path"], "subject")
        print(f"\n{ds_info['id']}: {len(subjects)} subjects found")

        for subject in subjects:
            print(f"  Processing sub-{subject} ...")
            epochs_array, metadata = get_epochs_for_subject(
                subject, dataset, ds_info["path"], ds_info["id"]
            )
            if epochs_array is None or len(epochs_array) == 0:
                continue

            fit_first = (rocket is None)
            features, rocket = embed(epochs_array, rocket=rocket, fit=fit_first)

            if fit_first:
                joblib.dump(rocket, ROCKET_PATH)
                print(f"    Saved fitted ROCKET to {ROCKET_PATH}")

            points = [
                PointStruct(
                    id=point_id + i,
                    vector=features[i].tolist(),
                    payload=metadata[i],
                )
                for i in range(len(features))
            ]
            qdrant.upsert(collection_name=COLLECTION, points=points)
            point_id += len(points)
            print(f"    Upserted {len(points)} epochs (total: {point_id})")

    print(f"\nDone. {point_id} epochs indexed in '{COLLECTION}'.")


if __name__ == "__main__":
    main()

# Example query usage:
# from ChromaExploration_4 import query_similar
# from qdrant_client import QdrantClient
# qdrant = QdrantClient(url=..., api_key=...)
# results = query_similar(my_epoch_array, qdrant)
# for hit in results[0]:
#     print(hit.score, hit.payload)
