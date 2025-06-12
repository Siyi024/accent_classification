# Accent Classification with ECAPA-TDNN

This project fine-tunes the ECAPA-TDNN model from [SpeechBrain](https://speechbrain.readthedocs.io/) to classify Mandarin- and Korean-accented English speech using the L2-ARCTIC dataset.

## 📁 Project Structure

- `train.json`, `valid.json`, `test.json` – data splits with accent labels
- `ecapa_finetune.py` – training script (based on SpeechBrain tutorial)
- `conf/` – YAML configuration files
- `classification_report.txt` – evaluation results
- `confusion_matrix.png` – confusion matrix visualization

## 🛠 Setup

```bash
pip install speechbrain torch torchaudio scikit-learn matplotlib

🚀 Run Training
python ecapa_finetune.py hparams/ecapa_l2arctic.yaml
📊 Results
	•	Overall Accuracy: 99.94%
	•	Perfect precision, recall, and F1-score for both Mandarin and Korean accents.

📦 Dataset

L2-ARCTIC corpus
Only .wav files and transcriptions are used. Audio is resampled to 16kHz.

📌 Notes
	•	Based on the SpeechBrain ECAPA-TDNN tutorial
	•	Adjusted data paths and labels for accent classification
	•	Runs on GPU (tested with NVIDIA A100)
