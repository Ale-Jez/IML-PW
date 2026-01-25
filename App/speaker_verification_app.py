"""
Speaker Verification Web Application
Frontend for speaker identification using ONNX model and H5 dataset
"""
import os
import io
import json
import tempfile
import numpy as np
import librosa
import onnxruntime as ort
import h5py
import yaml
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch

# =============================================================================
# CONFIGURATION
# =============================================================================
CHECKPOINT_DIR = "../checkpoints"
DATASET_PATH = "../outputs/logmels_fixed_split.h5"
ONNX_MODEL_PATH = "../checkpoints/speaker_classifier_backbone.onnx"

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {'wav', 'm4a', 'mp3', 'ogg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# SPEAKER ENROLLMENT (from H5 dataset)
# =============================================================================
def load_speaker_mapping():
    """Load speaker names and labels from H5 metadata"""
    try:
        with h5py.File(DATASET_PATH, "r") as f:
            if "speaker_mapping.yaml" in f["meta"]:
                mapping_yaml = f["meta"]["speaker_mapping.yaml"][()].decode("utf-8")
                speaker_mapping = yaml.safe_load(mapping_yaml)
                speakers_dict = speaker_mapping.get("speakers", {})
                # Reverse to get label -> name mapping
                label_to_name = {v: k for k, v in speakers_dict.items()}
                return label_to_name
    except Exception as e:
        print(f"Error loading speaker mapping: {e}")
    return {}

SPEAKER_MAPPING = load_speaker_mapping()
ENROLLED_SPEAKERS = set(SPEAKER_MAPPING.keys())  # Labels of enrolled speakers

# =============================================================================
# AUDIO PREPROCESSING
# =============================================================================
class AudioProcessor:
    def __init__(self, sr=16000, n_mels=64, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_audio(self, audio_data):
        """Load audio from bytes using temporary file"""
        try:
            # Write to temporary file for reliable loading
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                # Load with librosa
                y, sr = librosa.load(tmp_path, sr=self.sr, mono=True)
                return y
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")
    
    def compute_logmel(self, audio_samples, duration=3.0):
        """Compute log-mel spectrogram for a chunk"""
        # Take duration seconds
        chunk_samples = int(duration * self.sr)
        if len(audio_samples) > chunk_samples:
            audio_samples = audio_samples[:chunk_samples]
        elif len(audio_samples) < chunk_samples:
            # Pad with zeros if too short
            audio_samples = np.pad(audio_samples, (0, chunk_samples - len(audio_samples)))
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_samples, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Convert to log scale
        logmel = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize to [0, 1]
        logmel = (logmel + 80) / 80
        logmel = np.clip(logmel, 0, 1)
        
        return logmel  # Shape: (n_mels, time_steps)
    
    def process_audio(self, audio_data, duration=3.0):
        """Process audio file and return log-mel spectrogram"""
        audio_samples = self.load_audio(audio_data)
        
        # Check if audio is too short
        min_samples = int(1.0 * self.sr)  # At least 1 second
        if len(audio_samples) < min_samples:
            raise ValueError(f"Audio too short. Minimum 1 second required, got {len(audio_samples)/self.sr:.1f}s")
        
        logmel = self.compute_logmel(audio_samples, duration=duration)
        return logmel

audio_processor = AudioProcessor()

# =============================================================================
# ONNX INFERENCE
# =============================================================================
def load_onnx_session():
    """Load ONNX model session"""
    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")
    
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

onnx_session = load_onnx_session()

def infer_speaker(logmel_array):
    """
    Run inference on log-mel spectrogram using ONNX model
    
    Args:
        logmel_array: numpy array of shape (n_mels, time_steps)
    
    Returns:
        speaker_id: int (0-4)
        confidence: float (cosine similarity)
    """
    try:
        # Get ONNX input info
        input_info = onnx_session.get_inputs()
        output_info = onnx_session.get_outputs()
        
        print(f"ONNX Inputs: {[(inp.name, inp.shape, inp.type) for inp in input_info]}")
        print(f"ONNX Outputs: {[(out.name, out.shape, out.type) for out in output_info]}")
        
        # Prepare inputs based on what the model expects
        input_dict = {}
        
        for inp in input_info:
            if 'length' in inp.name.lower():
                # Lengths input - int64, shape [1]
                input_dict[inp.name] = np.array([logmel_array.shape[1]], dtype=np.int64)
            else:
                # Features input - should be float32
                # logmel_array is (n_mels, time_steps), need (1, 1, time_steps, n_mels)
                logmel_transposed = logmel_array.T  # (time_steps, n_mels)
                input_tensor = np.expand_dims(logmel_transposed, axis=(0, 1))  # (1, 1, T, F)
                input_tensor = input_tensor.astype(np.float32)
                input_dict[inp.name] = input_tensor
        
        print(f"Input dict: {[(k, v.shape, v.dtype) for k, v in input_dict.items()]}")
        
        # Run inference
        outputs = onnx_session.run(None, input_dict)
        
        # outputs[0] should be embeddings (1, 256)
        embedding = outputs[0][0]  # Shape: (256,)
        
        # Load speaker bank from checkpoint
        speaker_bank = load_speaker_bank()  # Shape: (5, 256)
        
        # Compute cosine similarities
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        similarities = embedding_norm @ speaker_bank.T  # (5,)
        
        speaker_id = int(np.argmax(similarities))
        confidence = float(similarities[speaker_id])
        
        print(f"Inference successful: speaker_id={speaker_id}, confidence={confidence}")
        
        return speaker_id, confidence
    
    except Exception as e:
        print(f"Inference error details: {str(e)}")
        raise ValueError(f"Inference failed: {e}")

def load_speaker_bank():
    """Load speaker embeddings bank from checkpoint"""
    try:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "train24", "best_model.pt")
        if not os.path.exists(checkpoint_path):
            # Try to find the latest checkpoint
            subdirs = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
            if subdirs:
                latest = sorted(subdirs, key=lambda x: int(x.replace('train', '')), reverse=True)[0]
                checkpoint_path = os.path.join(CHECKPOINT_DIR, latest, "best_model.pt")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract AAMSoftmax weights (speaker prototypes)
        if 'aamsm.weight' in checkpoint:
            speaker_bank = checkpoint['aamsm.weight'].cpu().numpy()
        else:
            raise KeyError("AAMSoftmax weights not found in checkpoint")
        
        # Normalize
        speaker_bank = speaker_bank / (np.linalg.norm(speaker_bank, axis=1, keepdims=True) + 1e-8)
        
        return speaker_bank
    
    except Exception as e:
        raise ValueError(f"Failed to load speaker bank: {e}")

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', speakers=SPEAKER_MAPPING)

@app.route('/api/verify', methods=['POST'])
def verify_speaker():
    """
    Verify speaker from uploaded audio file
    
    Returns:
        {
            'status': 'success' or 'error',
            'speaker_id': int,
            'speaker_name': str,
            'confidence': float,
            'is_enrolled': bool,
            'is_allowed': bool,
            'message': str
        }
    """
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No audio file selected'
            }), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Process audio
        logmel = audio_processor.process_audio(audio_data, duration=3.0)
        
        # Run inference
        speaker_id, confidence = infer_speaker(logmel)
        
        # Get speaker name
        speaker_name = SPEAKER_MAPPING.get(speaker_id, 'Unknown')
        
        # Check if speaker is enrolled
        is_enrolled = speaker_id in ENROLLED_SPEAKERS
        
        # Determine if allowed (enrolled + good confidence)
        is_allowed = is_enrolled and confidence > 0.3  # Threshold can be adjusted
        
        return jsonify({
            'status': 'success',
            'speaker_id': int(speaker_id),
            'speaker_name': speaker_name,
            'confidence': float(confidence),
            'is_enrolled': bool(is_enrolled),
            'is_allowed': bool(is_allowed),
            'message': f"Speaker identified: {speaker_name}" if is_allowed else f"Access denied. Unknown or unrecognized speaker."
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/speakers')
def get_speakers():
    """Get list of enrolled speakers"""
    return jsonify({
        'speakers': SPEAKER_MAPPING,
        'count': len(SPEAKER_MAPPING)
    }), 200

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'speaker_classifier',
        'version': '1.0',
        'enrolled_speakers': len(ENROLLED_SPEAKERS)
    }), 200

# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': f'File too large. Maximum {MAX_FILE_SIZE // 1024 // 1024}MB allowed'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("SPEAKER VERIFICATION APPLICATION")
    print("="*70)
    print(f"ONNX Model: {ONNX_MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Enrolled Speakers: {len(ENROLLED_SPEAKERS)}")
    print(f"Speaker Mapping: {SPEAKER_MAPPING}")
    print("\nStarting Flask server at http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
