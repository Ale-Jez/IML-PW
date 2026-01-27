# Speaker Verification Web Application

A web-based speaker identification and verification system using ONNX model and Flask.

## Features

- 🎤 **Live Audio Recording**: Record audio directly from your browser microphone
- 📁 **File Upload**: Upload pre-recorded audio files (WAV, MP3, M4A, OGG)
- 🎯 **Speaker Identification**: Identify speakers from a pool of 5 enrolled speakers
- ✅ **Access Control**: Determine if speaker is allowed based on enrollment and confidence
- 📊 **Real-time Results**: Get immediate feedback with confidence scores
- 🎨 **Modern UI**: Clean, responsive interface

## Architecture

```
speaker_verification_app.py (Flask Backend)
├── Audio Processing (Librosa)
├── ONNX Model Inference
├── Speaker Bank Loading
└── API Endpoints

templates/index.html (Web Frontend)
├── Audio Recording (Web Audio API)
├── File Upload (Drag & Drop)
├── Results Display
└── Responsive Design
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_app.txt
```

### 2. Prepare Files

Ensure you have:
- `./checkpoints/speaker_classifier_backbone.onnx` (ONNX model)
- `./checkpoints/train24/best_model.pt` (PyTorch checkpoint with speaker bank)
- `./outputs/logmels_fixed_split.h5` (Dataset with speaker metadata)

### 3. Run the Application

```bash
python speaker_verification_app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Open Browser**: Navigate to `http://localhost:5000`
2. **Choose Input**:
   - Click "Choose File" to upload audio
   - Click "Start Recording" to record live
3. **View Results**:
   - Speaker identification (name and ID)
   - Cosine similarity score
   - Access granted/denied status
   - Confidence visualization

## API Endpoints

### POST `/api/verify`
Verify speaker from audio file

**Request**:
```bash
curl -X POST -F "audio=@recording.wav" http://localhost:5000/api/verify
```

**Response**:
```json
{
  "status": "success",
  "speaker_id": 3,
  "speaker_name": "Piotr",
  "confidence": 0.342,
  "is_enrolled": true,
  "is_allowed": true,
  "message": "Speaker identified: Piotr"
}
```

### GET `/api/speakers`
Get list of enrolled speakers

**Response**:
```json
{
  "speakers": {
    "0": "Aleksander",
    "1": "Mantas",
    "2": "michał",
    "3": "Piotr",
    "4": "Rafał"
  },
  "count": 5
}
```

### GET `/api/health`
Health check endpoint

**Response**:
```json
{
  "status": "ok",
  "model": "speaker_classifier",
  "version": "1.0",
  "enrolled_speakers": 5
}
```

## Configuration

Edit `speaker_verification_app.py` to customize:

```python
CHECKPOINT_DIR = "./checkpoints"           # Path to model checkpoints
DATASET_PATH = "./outputs/logmels_fixed_split.h5"  # Dataset path
ONNX_MODEL_PATH = "./checkpoints/speaker_classifier_backbone.onnx"  # ONNX model
MAX_FILE_SIZE = 10 * 1024 * 1024          # Max upload size (10MB)
CONFIDENCE_THRESHOLD = 0.3                 # Minimum confidence for "allowed"
```

## How It Works

### 1. Audio Processing
- Loads audio file using librosa
- Resamples to 16kHz mono
- Computes log-mel spectrogram (64 bins, 3-second chunks)
- Normalizes to [0, 1] range

### 2. Inference
- Feeds spectrogram to ONNX model
- Model outputs 256-dimensional embedding
- Computes cosine similarity to 5 speaker prototypes
- Returns speaker ID with highest similarity

### 3. Verification
- Checks if speaker is enrolled (in training data)
- Compares confidence against threshold
- Returns access grant/deny decision

## Enrollment Requirements

**For a speaker to be "allowed":**
1. Speaker must be in the training set (enrolled)
2. Cosine similarity score must be > 0.3 (configurable)

Current enrolled speakers:
- 0: Aleksander
- 1: Mantas  
- 2: michał
- 3: Piotr
- 4: Rafał

## Troubleshooting

### "ONNX model not found"
- Ensure you've exported the model to ONNX format
- Check file path in `speaker_verification_app.py`

### "Audio too short"
- Minimum 1 second of audio required
- Maximum 10MB file size

### Microphone not working
- Check browser permissions (allow microphone access)
- Try uploading a file instead
- Use HTTPS in production (required for getUserMedia)

### Poor recognition accuracy
- Adjust `CONFIDENCE_THRESHOLD` in the app
- Try longer audio samples (3-5 seconds)
- Ensure audio quality is good (no background noise)

## Production Deployment

For production use:

1. **Use HTTPS**: Web Audio API requires HTTPS
2. **Add Authentication**: Implement user login
3. **Database**: Store verification logs
4. **Load Balancing**: Deploy multiple Flask instances
5. **Monitoring**: Add logging and alerts

Example with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 speaker_verification_app:app
```

## Performance

- **Inference Time**: ~200-500ms per 3-second chunk
- **Recording Latency**: <100ms
- **CPU Usage**: Minimal (ONNX Runtime optimized)

## Future Enhancements

- [ ] Real-time streaming inference
- [ ] Speaker enrollment UI
- [ ] Multi-speaker detection
- [ ] Liveness detection (anti-spoofing)
- [ ] Batch processing
- [ ] Performance metrics dashboard
- [ ] Database integration

## License

Internal use only

## Support

For issues or questions, contact the development team.
