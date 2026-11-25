# Skin Tone Analysis using OpenCV and MediaPipe Face Mesh

Skin tone detection from a single face photo. The pipeline locates facial skin regions with MediaPipe Face Mesh landmarks, corrects lighting with OpenCV, and classifies the skin tone using LAB color space, ITA angle, melanin index and the Fitzpatrick scale.

<p align="center">
  <img src="docs/flowchart_simple.svg" width="500" alt="Skin tone analysis flowchart">
</p>

## Deployment screenshots

### 1. Home page

Landing page.

<p align="center">
  <img src="docs/screenshots/01-home-page.png" width="850" alt="Home page">
</p>

### 2. Gender selection

Step 1 of the flow.

<p align="center">
  <img src="docs/screenshots/02-gender-selection.png" width="850" alt="Gender selection">
</p>

### 3. Capture or upload page

Webcam capture or image upload, with photo guidelines.

<p align="center">
  <img src="docs/screenshots/03-capture-or-upload.png" width="850" alt="Capture or upload page">
</p>

### 4. Test: uploading a group photo

Image with several people selected for upload.

<p align="center">
  <img src="docs/screenshots/04-upload-group-photo.png" width="850" alt="Test: uploading a group photo">
</p>

### 5. Validation error: multiple faces detected

Only one face per photo is accepted.

<p align="center">
  <img src="docs/screenshots/05-error-multiple-faces.png" width="850" alt="Validation error: multiple faces detected">
</p>

### 6. Test: uploading an image without a face

Non face image selected for upload.

<p align="center">
  <img src="docs/screenshots/06-upload-non-face-image.png" width="850" alt="Test: uploading an image without a face">
</p>

### 7. Validation error: no human face detected

Image is rejected before analysis.

<p align="center">
  <img src="docs/screenshots/07-error-no-face-detected.png" width="850" alt="Validation error: no human face detected">
</p>

### 8. Test: uploading a valid photo (example 1)

Single clear face selected for upload.

<p align="center">
  <img src="docs/screenshots/08-upload-example-1.png" width="850" alt="Test: uploading a valid photo (example 1)">
</p>

### 9. Face detection verification (example 1)

Detection confidence and face coverage check.

<p align="center">
  <img src="docs/screenshots/09-face-verification-example-1.png" width="850" alt="Face detection verification (example 1)">
</p>

### 10. Results: Very Light (Type I-II)

Privacy masked image, skin tone matrix and metrics.

<p align="center">
  <img src="docs/screenshots/10-results-example-1.png" width="850" alt="Results: Very Light (Type I-II)">
</p>

### 11. Test: uploading a valid photo (example 2)

Single clear face selected for upload.

<p align="center">
  <img src="docs/screenshots/11-upload-example-2.png" width="850" alt="Test: uploading a valid photo (example 2)">
</p>

### 12. Face detection verification (example 2)

Detection confidence and face coverage check.

<p align="center">
  <img src="docs/screenshots/12-face-verification-example-2.png" width="850" alt="Face detection verification (example 2)">
</p>

### 13. Results: Intermediate (Type III-IV)

Privacy masked image, skin tone matrix and metrics.

<p align="center">
  <img src="docs/screenshots/13-results-example-2.png" width="850" alt="Results: Intermediate (Type III-IV)">
</p>

### 14. Test: uploading a valid photo (example 3)

Single clear face selected for upload.

<p align="center">
  <img src="docs/screenshots/14-upload-example-3.png" width="850" alt="Test: uploading a valid photo (example 3)">
</p>

### 15. Face detection verification (example 3)

Detection confidence and face coverage check.

<p align="center">
  <img src="docs/screenshots/15-face-verification-example-3.png" width="850" alt="Face detection verification (example 3)">
</p>

### 16. Results: Brown (Type IV-V)

Privacy masked image, skin tone matrix and metrics.

<p align="center">
  <img src="docs/screenshots/16-results-example-3.png" width="850" alt="Results: Brown (Type IV-V)">
</p>

## Method

1. **Image validation:** size check (100 to 5000 px), low detail check, and exactly one face (MediaPipe Face Detection).
2. **Face verification:** MediaPipe Face Detection returns a bounding box and a confidence score, and the face must pass three checks before analysis continues.

   | Check | Formula | Accepted range |
   |---|---|---|
   | Detection confidence | `score` returned by MediaPipe Face Detection | at least 0.7 |
   | Face aspect ratio | `face_width / face_height` | 0.6 to 1.4 |
   | Face coverage | `(face_width x face_height) / (image_width x image_height)` | 5% to 80% |

   **Bounding box:** the pretrained MediaPipe detector predicts the rectangle around the face. It is returned as relative values between 0 and 1 (`xmin`, `ymin`, `width`, `height`), so it works for any image size. The values are converted to pixels:

   ```
   face_width  = bbox.width  x image_width
   face_height = bbox.height x image_height
   ```

   **Example:** for a 1000 x 800 image with a 300 x 360 px face box and a confidence of 0.92:
   - aspect ratio = 300 / 360 = 0.83, which passes
   - coverage = 108,000 / 800,000 = 13.5%, which passes
   - confidence 0.92 is at least 0.7, which passes

   A very low or very high aspect ratio means the box is not face shaped. A coverage below 5% means the face is too small to sample reliably, and above 80% it is too close to the camera. The verified face is then cropped with 50 px of padding, and a face mesh overlay is drawn for the verification page.
3. **Illumination correction:** gray world white balance, then CLAHE on the L channel in LAB space, blended 70% CLAHE with 30% original.
4. **Skin sampling:** 20 landmark points on the forehead (9) and both cheeks (11), each read as a 3x3 median, followed by IQR outlier removal and averaging to one RGB value.
5. **Metrics:** brightness, ITA angle and melanin index computed from the LAB values.
6. **Classification:** Fitzpatrick class from the rules below, plus undertone.
7. **Output:** detected RGB, class, metrics and an 8 shade skin tone matrix for the class.

## Classification criteria

Metric definitions:

| Metric | Formula |
|---|---|
| Brightness (L*) | L / 255 x 100 |
| ITA (degrees) | atan((L* - 50) / b*) |
| Melanin index | 100 - L* + 0.5 x a* |

a* and b* are measured relative to 128 (OpenCV LAB scale).

The rules are checked from top to bottom, and the first class where Rule A or Rule B is true is assigned.

| Fitzpatrick class | Rule A | or Rule B |
|---|---|---|
| Very Light (Type I-II) | Brightness > 70 and Melanin < 35 | ITA > 55 and Brightness > 65 |
| Light (Type II-III) | Brightness > 60 and Melanin < 45 | ITA > 41 and Brightness > 55 |
| Intermediate (Type III-IV) | Brightness > 45 and Melanin < 60 | ITA > 28 and Brightness > 40 |
| Brown (Type IV-V) | Brightness > 30 and Melanin < 75 | ITA > 10 and Brightness > 30 |
| Dark (Type V-VI) | None of the above | |

Undertone uses the raw a value: below 120 is Cool, 120 to 130 is Neutral, above 130 is Warm.

Only MediaPipe is a trained model. The classification is rule based image processing, so results are estimates that depend on lighting and camera quality. This is not a medical or dermatological tool.

## Privacy by design

- No name, email or account is collected, and only one face per photo is accepted.
- All processing runs locally with OpenCV and MediaPipe. No third party API or cloud vision service is called.
- Analysis uses the cropped face area, and the result depends only on the 20 sampled landmark points.
- The session keeps only derived values (RGB and metrics), never the photo itself.
- The preview shown to the user blurs everything except the sampled regions.
- Clicking **End Analysis** or **Start New Analysis** deletes all uploaded and processed images and clears the session. Every new upload also wipes the previous files.
- The `uploads/` and `static/results/` folders are excluded from Git.

## Tech stack

Python, OpenCV, MediaPipe, NumPy, Flask (used for the demo interface)

## Project structure

```
app.py               analysis pipeline and routes
requirements.txt
templates/           interface pages
static/css/
static/images/
docs/                diagrams
uploads/             temporary uploads (not tracked)
static/results/      temporary outputs (not tracked)
```

## Run locally

Requires Python 3.8 to 3.11 (MediaPipe 0.10.8).

```
git clone https://github.com/Rafa2208/skin-tone-detection-opencv-mediapipe.git
cd skin-tone-detection-opencv-mediapipe
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser. The webcam works only on localhost or HTTPS.

## Tips for good results

- Use natural daylight, facing a window, with the face looking straight ahead.
- Avoid harsh shadows, filters and heavy makeup.
- Only one face per photo.

## Limitations

- Temporary files use fixed names, so the demo is meant for one user at a time.
- If a user closes the browser without clicking End Analysis or Start New Analysis, their last images stay on the server until the next upload.
- Classification thresholds are fixed rules, not learned from a labelled dataset.
