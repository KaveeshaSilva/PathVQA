

## Process data
## Notice that 10-100 adaptive bottom-up attention features are used.

python3 tools/create_dictionary.py
python3 tools/compute_softscore.py
python3 tools/adaptive_detection_features_converter.py
