# AE-TM experiment archive

The 41.8 MB experiment package is stored as 70 Git blobs because the connected GitHub interface does not expose Release Asset upload. It contains generated data, 24 selected checkpoints, training/evaluation logs, output images, runtime metadata, and deployment parity records.

After cloning this branch run `python3 artifacts/reconstruct_archive.py`. The script concatenates the parts and verifies byte count and SHA-256. See `docs/capture_tm/JOINT_RESULTS.md` for the report.
