# Song Library Classifier

This project compares two music libraries using audio embeddings. It creates one MERT embedding per matched iTunes preview, then stores the final embedding archive alongside the labeled song catalog.

## Included data

- `jacksfulllibrary.csv` and `sinclairsfulllibrary.csv`: original library exports.
- `combined_music_library.csv`: recording-level catalog joined by exact ISRC and labeled `Jack`, `Sinclair`, or `Shared`.
- `embeddings/song_embeddings.npz`: final MERT embeddings. The archive contains `isrc` and a `(2183, 768)` float32 embedding matrix.
- `embeddings/song_embedding_manifest.csv`: row-level embedding status and iTunes match metadata for all 2,383 catalog records.

The combined catalog has 11 exact-ISRC shared recordings. Embeddings were created for 2,183 recordings; 200 could not be embedded because the public iTunes search did not yield a confident preview match.

## Reproducing embedding generation

Create a Python environment, install the requirements, then run:

```bash
python generate_embeddings.py
```

The script reads `combined_music_library.csv`, searches the public iTunes Search API for a 30-second preview, decodes audio at 24 kHz, and mean-pools the final hidden states of [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M). It requires internet access to retrieve iTunes previews and download the pretrained MERT model on first run.

The generator creates resumable intermediate vectors and attempt logs inside `embeddings/` while it runs. Those files are ignored by Git; the submitted archive and manifest are the final outputs.

## Repository policy

No audio previews are stored in this repository. The source catalog IDs, final embeddings, manifest, and generation scripts are retained so the analysis can be inspected and reproduced.
