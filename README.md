# Voice Emotion Recognition - CNN

A demonstrable speech-emotion classification application: the browser records or uploads audio, the API creates and displays a log-Mel spectrogram, and a CNN returns an emotion class and its probabilities.

## Product decision

The **local web application** is the primary deliverable because it demonstrates the complete pipeline: voice capture -> preprocessing -> CNN -> prediction. A collaborative document should accompany the repository only for the report, experimental decisions, results, and the demonstration-video link. It does not replace the application because it cannot record audio or run the CNN reproducibly.

## Repository structure

```text
app/                API, log-Mel feature extraction, and two PyTorch CNNs
web/                Static page using MediaRecorder
data/manifest.csv   path,label,speaker; enables speaker-disjoint splitting
artifacts/          Generated weights and labels (not versioned)
train.py            Scratch or transfer training with speaker-independent evaluation
```

## Required architectures

- **`scratch`:** four blocks of `Conv2d 3x3 -> BatchNorm -> ReLU -> Conv2d 3x3 -> BatchNorm -> ReLU -> MaxPool2d 2x2`, with 32, 64, 128, and 256 channels. Global average pooling, dropout, and a final linear layer classify the emotions. During training, frequency/time masks (SpecAugment) and random temporal shifts are applied.
- **`transfer`:** an ImageNet-pretrained ResNet-18. Each log-Mel spectrogram is repeated across three RGB channels; every backbone weight is frozen and only the final `fc` layer is optimized.

## Running the project

Prerequisites: Python 3.10+ and `ffmpeg` on the PATH (required to process WebM recordings from the browser).

On Ubuntu or Debian, install it with:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_emodb.py
# The script reads data/emodb/wav and creates manifest.csv with path,label,speaker.
python train.py --architecture scratch --epochs 25
python train.py --architecture transfer --epochs 25
uvicorn app.main:app --reload
```

Run the structural validation before the first training run:

```powershell
pytest -q
```

Open `http://127.0.0.1:8000`. Use a public dataset such as EmoDB, RAVDESS, or CREMA-D. Populate `data/manifest.csv` with `path,label,speaker`; `speaker` is mandatory to ensure that a person's voice never appears in both training and testing. Do not commit the dataset or large weights: provide them through a licensed link and reproducible instructions.

For inference, the page accepts WebM, WAV, MP3, FLAC, OGG, and MP4 files with an audio track. The application analyzes the first four seconds of each file; short clips are zero-padded.

## Development plan and delivery criteria

1. **Data:** choose one public corpus (recommended: RAVDESS, because it provides actor IDs), normalize its emotion labels, and generate `manifest.csv`.
2. **Scratch CNN:** run `--architecture scratch`; record accuracy and macro-F1 on never-seen speakers.
3. **Transfer learning:** run `--architecture transfer`, which uses ImageNet-pretrained ResNet-18. The spectrogram is repeated in RGB, the backbone remains frozen, and only the final layer is trained. Compare both CNNs fairly using the same split.
4. **App and demo:** select either trained model in the page's **Model** menu, then test recording -> spectrogram -> prediction. The API loads `artifacts/model-scratch.pt` or `artifacts/model-transfer.pt` according to that selection; no manual copy to `model.pt` is required.
5. **Deployment and package:** deploy the API, test the microphone at the public URL, and add results, contributions, and limitations to this README.

## Results

Both models were evaluated on the same speaker-disjoint holdout split (test speakers `08` and `15`). No speaker appears in both the training and test sets.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Scratch CNN | 57.0% | 49.5% |
| Transfer learning (frozen ResNet-18) | 64.0% | 61.2% |

The transfer-learning model achieved the best results and is the default model in the web interface. The interface also lets the user run either trained model for comparison.

## Responsible-use limitations

The inferred emotion is an estimate of acoustic patterns learned from the training dataset, not a psychological diagnosis or a reliable personal attribute. Use only authorized audio and state this limitation explicitly in the presentation.

## Team and contributions

Complete this section before delivery with names and verifiable contributions:

| Person | Planned contribution |
|---|---|
| Diego Magno MONTANS ZUCHERAM | Data preparation and speaker-independent evaluation |
| Gustavo KURATOMI PEREIRA E COSTA | Scratch CNN and transfer-learning implementation |
| Thiago ROSA CARVALHO | Web application, deployment, and documentation |

## Submission checklist

- [ ] Both CNNs have been trained: one from scratch and one using transfer learning
- [ ] The scratch CNN has four double 3x3 blocks, BatchNorm, ReLU, max pooling, and SpecAugment
- [ ] The transfer CNN uses a frozen ResNet-18 and RGB-repeated spectrograms
- [ ] A clean environment reproduces installation, training, and execution
- [ ] `results-scratch.json` and `results-transfer.json` demonstrate speaker-disjoint splitting and macro-F1
- [ ] Dataset, weights, and credentials were not committed to Git
- [ ] The demo was tested in a browser with microphone permission
- [ ] Repository URL and public app URL are ready to submit on 2026-09-29
