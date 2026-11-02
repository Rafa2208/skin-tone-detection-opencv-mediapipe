from flask import Flask, render_template, request, jsonify, session, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
import os
import base64
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Every file this app writes for a user. All of them are wiped by clear_user_data()
USER_FILES = [
    os.path.join('uploads', 'original.jpg'),
    os.path.join('static', 'results', 'overlay.jpg'),
    os.path.join('static', 'results', 'cropped.jpg'),
    os.path.join('static', 'results', 'privacy_masked.jpg'),
]

# Face regions used for skin sampling
LEFT_CHEEK = [330, 280, 411, 352, 345]
RIGHT_CHEEK = [101, 50, 187, 205, 123, 116]
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454]

# ============================================
# PRIVACY: DATA CLEANUP
# ============================================

def clear_user_data():
    """Delete all uploaded and analysed images and reset the session."""
    for path in USER_FILES:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"Could not delete {path}: {e}")
    session.clear()

# ============================================
# SKIN TONE ANALYSIS FUNCTIONS
# ============================================

def advanced_white_balance(img):
    """Advanced white balance correction"""
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / (avg_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / (avg_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / (avg_r + 1e-6)), 0, 255)

    return result.astype(np.uint8)

def adaptive_illumination_correction(img):
    """Adaptive illumination correction"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l)
    l_final = cv2.addWeighted(l, 0.3, l_corrected, 0.7, 0)
    lab_corrected = cv2.merge([l_final, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

def validate_image_content(image):
    """Validate if image contains a proper human face"""
    if image is None:
        return False, "Could not read the image"

    h, w, _ = image.shape

    if w < 100 or h < 100 or w > 5000 or h > 5000:
        return False, "Image size must be between 100x100 and 5000x5000 pixels"

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    color_variance = np.var(pixels, axis=0).mean()

    if color_variance < 100:
        return False, "Image appears to be of nature, objects, or has low detail"

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.6
    )

    results = face_detection.process(image_rgb)

    if not results.detections:
        return False, "No human face detected"

    if len(results.detections) > 1:
        return False, f"Multiple faces detected ({len(results.detections)} faces). Please upload an image with only one face"

    return True, "Valid face detected"

def detect_and_verify_face(image):
    """Detect face and return verification data"""
    h, w, _ = image.shape

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.7,
        refine_landmarks=True
    )

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.7
    )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    results_detection = face_detection.process(image_rgb)

    if not results_mesh.multi_face_landmarks or not results_detection.detections:
        return None, "No face detected"

    if len(results_mesh.multi_face_landmarks) > 1:
        return None, "Multiple faces detected"

    face_landmarks = results_mesh.multi_face_landmarks[0]
    detection = results_detection.detections[0]

    detection_confidence = detection.score[0]
    if detection_confidence < 0.7:
        return None, f"Low face detection confidence: {detection_confidence:.2f}"

    bbox = detection.location_data.relative_bounding_box
    face_width = bbox.width * w
    face_height = bbox.height * h
    aspect_ratio = face_width / (face_height + 1e-6)

    if aspect_ratio < 0.6 or aspect_ratio > 1.4:
        return None, "Unusual face proportions detected"

    face_area = face_width * face_height
    image_area = w * h
    face_coverage = face_area / image_area

    if face_coverage < 0.05:
        return None, "Face is too small in the image"

    if face_coverage > 0.8:
        return None, "Face is too large in the image"

    # Create visualization
    overlay_image = draw_face_mesh_overlay(image.copy(), face_landmarks, w, h)

    # Draw bounding box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)

    padding = 50
    x_padded = max(0, x - padding)
    y_padded = max(0, y - padding)
    width_padded = min(w - x_padded, width + 2 * padding)
    height_padded = min(h - y_padded, height + 2 * padding)

    cv2.rectangle(overlay_image, (x_padded, y_padded),
                  (x_padded + width_padded, y_padded + height_padded),
                  (0, 255, 0), 3)

    # Crop face
    cropped_image = image[y_padded:y_padded + height_padded,
                          x_padded:x_padded + width_padded]

    return {
        'overlay_image': overlay_image,
        'cropped_image': cropped_image,
        'face_landmarks': face_landmarks,
        'confidence': detection_confidence,
        'face_coverage': face_coverage,
        'aspect_ratio': aspect_ratio
    }, None

def draw_face_mesh_overlay(image, face_landmarks, w, h):
    """Draw face mesh overlay on image"""
    overlay = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    connection_spec = mp_drawing.DrawingSpec(thickness=1, color=(255, 0, 255))

    mp_drawing.draw_landmarks(
        image=overlay,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )

    regions = [
        (LEFT_CHEEK, (0, 255, 255)),
        (RIGHT_CHEEK, (0, 255, 255)),
        (FOREHEAD, (255, 255, 0))
    ]

    for indices, color in regions:
        coords = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x = max(0, min(w - 1, int(lm.x * w)))
            y = max(0, min(h - 1, int(lm.y * h)))
            coords.append([x, y])

        if coords:
            cv2.fillPoly(overlay, [np.array(coords, dtype=np.int32)], color)

    alpha = 0.4
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def extract_skin_samples_filtered(image, face_landmarks, w, h):
    """Extract skin samples with outlier filtering"""
    all_rgb_values = []

    for indices in [LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD]:
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x = max(0, min(w - 1, int(lm.x * w)))
            y = max(0, min(h - 1, int(lm.y * h)))

            region_samples = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        pixel = image[ny, nx]
                        region_samples.append((int(pixel[2]), int(pixel[1]), int(pixel[0])))

            if region_samples:
                median_r = int(np.median([rgb[0] for rgb in region_samples]))
                median_g = int(np.median([rgb[1] for rgb in region_samples]))
                median_b = int(np.median([rgb[2] for rgb in region_samples]))
                all_rgb_values.append((median_r, median_g, median_b))

    if not all_rgb_values:
        return None

    r_vals = [rgb[0] for rgb in all_rgb_values]
    g_vals = [rgb[1] for rgb in all_rgb_values]
    b_vals = [rgb[2] for rgb in all_rgb_values]

    def remove_outliers(values):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [v for v in values if lower_bound <= v <= upper_bound]

    r_filtered = remove_outliers(r_vals)
    g_filtered = remove_outliers(g_vals)
    b_filtered = remove_outliers(b_vals)

    avg_r = int(np.mean(r_filtered)) if r_filtered else int(np.mean(r_vals))
    avg_g = int(np.mean(g_filtered)) if g_filtered else int(np.mean(g_vals))
    avg_b = int(np.mean(b_filtered)) if b_filtered else int(np.mean(b_vals))

    return avg_r, avg_g, avg_b

def classify_skin_tone_robust(r, g, b, l_val, a_val, b_val, h_val, s_val, v_val):
    """Multi-factor skin tone classification"""
    l_scaled = (l_val / 255.0) * 100
    a_scaled = a_val - 128
    b_scaled = b_val - 128

    ita = np.degrees(np.arctan((l_scaled - 50) / (b_scaled + 0.001)))
    brightness = l_scaled
    redness = a_scaled

    melanin_index = 100 - brightness + (redness * 0.5)

    if (brightness > 70 and melanin_index < 35) or (ita > 55 and brightness > 65):
        category = "Very Light (Type I-II)"
    elif (brightness > 60 and melanin_index < 45) or (ita > 41 and brightness > 55):
        category = "Light (Type II-III)"
    elif (brightness > 45 and melanin_index < 60) or (ita > 28 and brightness > 40):
        category = "Intermediate (Type III-IV)"
    elif (brightness > 30 and melanin_index < 75) or (ita > 10 and brightness > 30):
        category = "Brown (Type IV-V)"
    else:
        category = "Dark (Type V-VI)"

    return category, ita, brightness, melanin_index

def create_skin_tone_palette_realistic(category):
    """Predefined 8 shade skin tone matrix for each Fitzpatrick category"""
    skin_palettes = {
        "Very Light (Type I-II)": [
            [255, 240, 245], [255, 230, 230], [255, 220, 210], [250, 210, 200],
            [245, 200, 190], [240, 190, 180], [235, 185, 175], [230, 180, 170]
        ],
        "Light (Type II-III)": [
            [240, 200, 180], [235, 190, 170], [230, 185, 165], [225, 180, 160],
            [220, 175, 155], [215, 170, 150], [210, 165, 145], [205, 160, 140]
        ],
        "Intermediate (Type III-IV)": [
            [200, 160, 140], [195, 155, 135], [190, 150, 130], [185, 145, 125],
            [180, 140, 120], [175, 135, 115], [170, 130, 110], [165, 125, 105]
        ],
        "Brown (Type IV-V)": [
            [160, 120, 100], [155, 115, 95], [150, 110, 90], [145, 105, 85],
            [140, 100, 80], [135, 95, 75], [130, 90, 70], [125, 85, 65]
        ],
        "Dark (Type V-VI)": [
            [120, 80, 60], [115, 75, 55], [110, 70, 50], [105, 65, 45],
            [100, 60, 40], [95, 55, 35], [90, 50, 30], [85, 45, 25]
        ]
    }
    return skin_palettes.get(category, skin_palettes["Intermediate (Type III-IV)"])

def create_privacy_mask(image, face_landmarks, w, h):
    """Create a privacy-masked image showing only analysis regions"""
    mask = np.zeros((h, w), dtype=np.uint8)

    for indices in [LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD]:
        coords = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x = max(0, min(w - 1, int(lm.x * w)))
            y = max(0, min(h - 1, int(lm.y * h)))
            coords.append([x, y])

        if coords:
            cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 255)

    blurred = cv2.GaussianBlur(image, (99, 99), 30)
    privacy_image = blurred.copy()
    privacy_image[mask == 255] = image[mask == 255]
    return privacy_image

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def index():
    """Landing page with logo"""
    return render_template('index.html')

@app.route('/gender')
def gender_page():
    """Gender selection page"""
    return render_template('gender.html')

@app.route('/capture')
def capture_page():
    """Image capture/upload page"""
    gender = request.args.get('gender', 'male')
    session['gender'] = gender
    return render_template('capture.html', gender=gender)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload"""
    try:
        # Remove anything left from a previous attempt, keep the chosen gender
        gender = session.get('gender', 'male')
        clear_user_data()
        session['gender'] = gender

        if request.is_json and 'image' in request.json:
            image_data = request.json['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif 'file' in request.files:
            file = request.files['file']
            nparr = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400

        is_valid, message = validate_image_content(image)
        if not is_valid:
            return jsonify({'error': message}), 400

        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg'), image)

        result, error = detect_and_verify_face(image)
        if error:
            clear_user_data()
            session['gender'] = gender
            return jsonify({'error': error}), 400

        overlay_path = os.path.join('static', 'results', 'overlay.jpg')
        cropped_path = os.path.join('static', 'results', 'cropped.jpg')

        cv2.imwrite(overlay_path, result['overlay_image'])
        cv2.imwrite(cropped_path, result['cropped_image'])

        session['cropped_path'] = cropped_path
        session['confidence'] = float(result['confidence'])
        session['face_coverage'] = float(result['face_coverage'])

        return jsonify({
            'success': True,
            'overlay_image': f'/static/results/overlay.jpg?t={os.urandom(4).hex()}',
            'confidence': f"{result['confidence'] * 100:.1f}",
            'coverage': f"{result['face_coverage'] * 100:.1f}"
        })

    except Exception as e:
        print(f"ERROR in upload_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/verify')
def verify_page():
    """Face verification page"""
    return render_template('verify.html')

@app.route('/get_verification_data')
def get_verification_data():
    """Return verification data from session"""
    try:
        confidence = session.get('confidence', None)
        face_coverage = session.get('face_coverage', None)
        gender = session.get('gender', 'male')

        if confidence is None or face_coverage is None:
            return jsonify({
                'success': False,
                'error': 'No verification data in session',
                'confidence': '0.0',
                'coverage': '0.0',
                'gender': gender
            })

        return jsonify({
            'success': True,
            'confidence': f"{float(confidence) * 100:.1f}",
            'coverage': f"{float(face_coverage) * 100:.1f}",
            'gender': gender
        })
    except Exception as e:
        print(f"ERROR in get_verification_data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_skin():
    """Analyze skin tone"""
    try:
        cropped_path = session.get('cropped_path')
        if not cropped_path or not os.path.exists(cropped_path):
            return jsonify({'error': 'No image to analyze'}), 400

        image = cv2.imread(cropped_path)
        image = advanced_white_balance(image)
        image = adaptive_illumination_correction(image)

        h, w, _ = image.shape

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.7,
            refine_landmarks=True
        )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return jsonify({'error': 'Face landmarks not detected'}), 400

        face_landmarks = results.multi_face_landmarks[0]

        # Privacy masked image: blurred except the analysis regions
        privacy_masked_image = create_privacy_mask(image, face_landmarks, w, h)
        cv2.imwrite(os.path.join('static', 'results', 'privacy_masked.jpg'), privacy_masked_image)

        skin_rgb = extract_skin_samples_filtered(image, face_landmarks, w, h)
        if skin_rgb is None:
            return jsonify({'error': 'Failed to extract skin samples'}), 400

        avg_r, avg_g, avg_b = skin_rgb

        skin_lab = cv2.cvtColor(np.uint8([[skin_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
        skin_hsv = cv2.cvtColor(np.uint8([[skin_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        l_val, a_val, b_val = skin_lab
        h_val, s_val, v_val = skin_hsv

        category, ita, brightness, melanin_index = classify_skin_tone_robust(
            avg_r, avg_g, avg_b, l_val, a_val, b_val, h_val, s_val, v_val
        )

        if a_val > 130:
            undertone = "Warm"
        elif a_val < 120:
            undertone = "Cool"
        else:
            undertone = "Neutral"

        skin_tone_palette = create_skin_tone_palette_realistic(category)

        session['skin_rgb'] = (int(avg_r), int(avg_g), int(avg_b))
        session['category'] = category
        session['ita'] = float(ita)
        session['brightness'] = float(brightness)
        session['melanin_index'] = float(melanin_index)
        session['undertone'] = undertone
        session['skin_tone_palette'] = skin_tone_palette

        return jsonify({'success': True})

    except Exception as e:
        print(f"ERROR in analyze_skin: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results_page():
    """Skin tone results page"""
    return render_template('results.html')

@app.route('/get_results')
def get_results():
    """Return stored analysis results for the results page"""
    if 'category' not in session:
        return jsonify({'error': 'No analysis found. Please start again.'}), 400

    r, g, b = session['skin_rgb']
    return jsonify({
        'success': True,
        'category': session['category'],
        'skin_rgb': {'r': r, 'g': g, 'b': b},
        'ita': session['ita'],
        'brightness': session['brightness'],
        'melanin_index': session['melanin_index'],
        'undertone': session['undertone'],
        'skin_tone_palette': session['skin_tone_palette']
    })

@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Delete uploaded and analysed images, used by End Analysis and Start New Analysis"""
    clear_user_data()
    return jsonify({'success': True})

@app.route('/end')
def end_page():
    """Final page shown after End Analysis"""
    return render_template('end.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("=" * 60)
    print("TonePrint - AI Skin Tone Analysis")
    print("=" * 60)
    print("Access the app at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
