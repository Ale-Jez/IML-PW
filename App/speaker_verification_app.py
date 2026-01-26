"""
Speaker Verification Web Application - Logmel Statistics Based
"""
import os
import tempfile
import sys
import numpy as np
import librosa
import h5py
import yaml
from flask import Flask, render_template, request, jsonify

# =============================================================================
# CONFIGURATION
# =============================================================================
H5_PATH = "../outputs/logmels_spkid_aug_26-01-25_19-04-46.h5"
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {'wav', 'm4a', 'mp3', 'ogg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# SPEAKER ENROLLMENT (from H5 dataset)
# =============================================================================
def load_speaker_mapping():
    """Load speaker names and labels from H5 metadata"""
    try:
        with h5py.File(H5_PATH, "r") as f:
            if "speaker_mapping.yaml" in f["meta"]:
                mapping_yaml = f["meta"]["speaker_mapping.yaml"][()].decode("utf-8")
                speaker_mapping = yaml.safe_load(mapping_yaml)
                speakers_dict = speaker_mapping.get("speakers", {})
                label_to_name = {v: k for k, v in speakers_dict.items()}
                return label_to_name
    except Exception as e:
        print(f"Error loading speaker mapping: {e}")
    return {}

SPEAKER_MAPPING = load_speaker_mapping()
ENROLLED_SPEAKERS = set(SPEAKER_MAPPING.keys())

# =============================================================================
# SPEAKER EMBEDDINGS (from H5 training data)
# =============================================================================
def load_speaker_embeddings():
    """Load speaker embeddings from H5 training data"""
    try:
        embeddings_by_speaker = {}
        speaker_mapping = load_speaker_mapping()
        
        if not os.path.exists(H5_PATH):
            print(f"Warning: H5 file not found at {H5_PATH}")
            return embeddings_by_speaker
        
        with h5py.File(H5_PATH, 'r') as f:
            train_logmels = f['train']['logmel'][:]
            train_labels = f['train']['label'][:]
            
            for label in sorted(set(train_labels)):
                if label not in speaker_mapping:
                    continue
                
                mask = train_labels == label
                speaker_logmels = train_logmels[mask]
                
                mean_embeddings = []
                std_embeddings = []
                
                for logmel in speaker_logmels:
                    mean_embeddings.append(logmel.mean(axis=1))
                    std_embeddings.append(logmel.std(axis=1))
                
                mean_emb = np.mean(mean_embeddings, axis=0)
                std_emb = np.mean(std_embeddings, axis=0)
                
                full_embedding = np.concatenate([mean_emb, std_emb])
                full_embedding = full_embedding / (np.linalg.norm(full_embedding) + 1e-8)
                
                speaker_name = speaker_mapping[label]
                embeddings_by_speaker[label] = {
                    "name": speaker_name,
                    "embedding": full_embedding.astype(np.float32)
                }
                print(f"✓ Loaded embedding for {speaker_name} (Speaker {label})")
        
        return embeddings_by_speaker
    
    except Exception as e:
        print(f"Failed to load speaker embeddings: {e}")
        return {}

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
        """Load audio from bytes"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                y, sr = librosa.load(tmp_path, sr=self.sr, mono=True)
                return y
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")
    
    def compute_logmel(self, audio_samples, duration=3.0):
        """Compute log-mel spectrogram (match training preprocessing)"""
        chunk_samples = int(duration * self.sr)
        if len(audio_samples) > chunk_samples:
            audio_samples = audio_samples[:chunk_samples]
        elif len(audio_samples) < chunk_samples:
            audio_samples = np.pad(audio_samples, (0, chunk_samples - len(audio_samples)))
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_samples, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        logmel = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
        
        min_time_steps = 50
        if logmel.shape[1] < min_time_steps:
            logmel = np.pad(logmel, ((0, 0), (0, min_time_steps - logmel.shape[1])))
        
        return logmel
    
    def process_audio(self, audio_data, duration=3.0):
        """Process audio file"""
        audio_samples = self.load_audio(audio_data)
        
        min_samples = int(1.0 * self.sr)
        if len(audio_samples) < min_samples:
            raise ValueError(f"Audio too short. Minimum 1 second required, got {len(audio_samples)/self.sr:.1f}s")
        
        logmel = self.compute_logmel(audio_samples, duration=duration)
        return logmel

print("Loading speaker embeddings from dataset...")
SPEAKER_EMBEDDINGS = load_speaker_embeddings()
print(f"✓ Loaded embeddings for {len(SPEAKER_EMBEDDINGS)} speakers")

audio_processor = AudioProcessor()

# =============================================================================
# SPEAKER VERIFICATION INFERENCE
# =============================================================================
def infer_speaker(logmel_array, threshold=0.5):
    """Verify speaker by logmel statistics comparison"""
    try:
        test_mean = logmel_array.mean(axis=1)
        test_std = logmel_array.std(axis=1)
        test_embedding = np.concatenate([test_mean, test_std])
        test_embedding = test_embedding / (np.linalg.norm(test_embedding) + 1e-8)
        
        similarities = {}
        for speaker_id, speaker_data in SPEAKER_EMBEDDINGS.items():
            template = speaker_data['embedding']
            similarity = np.dot(test_embedding, template)
            similarities[speaker_id] = similarity
        
        speaker_id = max(similarities, key=similarities.get)
        confidence = similarities[speaker_id]
        is_verified = confidence > threshold
        
        print(f"  All similarities: {[f'{similarities[i]:.4f}' for i in sorted(similarities.keys())]}")
        print(f"  Best match: Speaker {speaker_id} ({SPEAKER_MAPPING.get(speaker_id, 'Unknown')}), Confidence: {confidence:.4f}")
        
        return speaker_id, confidence, is_verified
    
    except Exception as e:
        print(f"Inference error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Inference failed: {e}")

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html', speakers=SPEAKER_MAPPING)

@app.route('/api/verify', methods=['POST'])
def verify_speaker():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'status': 'error', 'message': 'No audio file selected'}), 400
        
        audio_data = audio_file.read()
        logmel = audio_processor.process_audio(audio_data, duration=3.0)
        
        speaker_id, confidence, is_verified = infer_speaker(logmel, threshold=0.5)
        
        speaker_name = SPEAKER_MAPPING.get(speaker_id, 'Unknown')
        is_enrolled = speaker_id in ENROLLED_SPEAKERS
        is_allowed = is_verified and is_enrolled
        
        return jsonify({
            'status': 'success',
            'speaker_id': int(speaker_id),
            'speaker_name': speaker_name,
            'confidence': float(confidence),
            'is_enrolled': bool(is_enrolled),
            'is_allowed': bool(is_allowed),
            'message': f"Speaker identified: {speaker_name}" if is_allowed else "Access denied."
        }), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/speakers')
def get_speakers():
    return jsonify({'speakers': SPEAKER_MAPPING, 'count': len(SPEAKER_MAPPING)}), 200

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'ok', 'enrolled_speakers': len(ENROLLED_SPEAKERS)}), 200

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("SPEAKER VERIFICATION APPLICATION (H5-BASED)")
    print("="*70)
    print(f"Dataset: {H5_PATH}")
    print(f"Enrolled Speakers: {len(ENROLLED_SPEAKERS)}")
    print(f"Speaker Mapping: {SPEAKER_MAPPING}")
    print("\nStarting Flask server at http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
