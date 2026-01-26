import h5py
import numpy as np

f = h5py.File('outputs/logmels_fixed_split.h5', 'r')
print('train/logmel shape:', f['train/logmel'].shape)
print('train/label shape:', f['train/label'].shape) 
print('Unique labels:', sorted(set(f['train/label'][:])))
print('train/logmel dtype:', f['train/logmel'].dtype)

# Check speaker mapping
if 'meta' in f and 'speaker_mapping.yaml' in f['meta']:
    import yaml
    mapping_yaml = f["meta"]["speaker_mapping.yaml"][()].decode("utf-8")
    speaker_mapping = yaml.safe_load(mapping_yaml)
    print('Speaker mapping:', speaker_mapping)

f.close()
