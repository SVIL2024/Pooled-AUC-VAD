"""Cross-domain WSVAD research package (Stage 0).

Conventions inherited from VadCLIP / frequenceVad:
  * CLIP ViT-B/16 frame features, one .npy per video, shape (N, 512), float16
  * CLIP stride = 16 raw video frames per feature
  * frame-level GT is a flat float32 array over all test videos concatenated,
    each video contributing N*16 entries
"""

__all__ = ["data", "metrics"]