

def parse_crema_label(filename):
    parts = filename.split('_')
    label_map = {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
        'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
    }
    return label_map.get(parts[2], 'unknown')

def parse_savee_label(filename):
    label_map = {
        'a': 'angry', 'd': 'disgust', 'f': 'fear',
        'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise'
    }
    label_code = filename[:2] if filename[:2] == 'sa' else filename[0]
    return label_map.get(label_code, 'unknown')

def parse_tess_label(filename):
    parts = filename.split('_')
    label = parts[2].replace('.wav', '').lower()
    return label
