"""Split, NDVI baseline, segmentation metrics."""
import numpy as np


def split_chips(chips, train_frac=0.60, val_frac=0.20):
    """Split by column position. Contiguous blocks, not random - see README."""
    cols = sorted({c["col"] for c in chips})
    # 60% train, 20% val, 20% test
    a = cols[int(train_frac * len(cols))]
    b = cols[int((train_frac + val_frac) * len(cols))]

    train = [c for c in chips if c["col"] < a]
    val = [c for c in chips if a <= c["col"] < b]
    test = [c for c in chips if c["col"] >= b]

    print(f"train {len(train)} / val {len(val)} / test {len(test)}")
    return train, val, test

def ndvi(img):
    """img is (4, H, W) uint8, NAIP band order R G B NIR."""
    # cast to float32 to avoid overflow when computing nir-red
    red = img[0].astype("float32")
    nir = img[3].astype("float32")
    # compute NDVI, add small value to avoid divide by zero for pixels with no vegetation (nir=red=0)
    return (nir - red) / (nir + red + 1e-6)

def metrics(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)

    tp = (pred & truth).sum()
    fp = (pred & ~truth).sum()
    fn = (~pred & truth).sum()

    # precision is the fraction of predicted positives that are true positives
    precision = tp / (tp + fp + 1e-9)
    # recall is the fraction of true positives that are predicted positives
    recall = tp / (tp + fn + 1e-9)

    return {
        # IOU is the intersection over union of predicted and true positives
        "iou": tp / (tp + fp + fn + 1e-9),
        "precision": precision,
        "recall": recall,
        # F1 score is the harmonic mean of precision and recall
        "f1": 2 * precision * recall / (precision + recall + 1e-9),
    }

def tune_threshold(chips, lo=-0.10, hi=0.60, step=0.02):
    """Best-F1 NDVI cutoff. Train chips only."""
    nd = np.concatenate([ndvi(c["img"]).ravel() for c in chips])
    truth = np.concatenate([c["msk"].ravel() for c in chips])

    best_t, best_f1 = None, -1.0
    for t in np.arange(lo, hi, step):
        f1 = metrics(nd > t, truth)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    print(f"threshold {best_t:.2f}, train F1 {best_f1:.3f}")
    return best_t