# SkinAura: AI-Powered Skin Type Detection & Cosmetics Recommendation System

Complete source code for the SkinAura project — a CNN-based prototype that
predicts facial skin type (**oily / dry / normal**) from a photo and gives
basic cosmetics recommendations, exactly matching the block diagram:

```
Dataset -> Image Preprocessing -> CNN Classification -> Skin Type Prediction
        -> Cosmetics Recommendation (Product A / Product B) -> Final Output
```

## Project Files

| File               | Purpose                                                        |
|--------------------|-----------------------------------------------------------------|
| `config.py`        | All paths and hyperparameters in one place                     |
| `model.py`         | The CNN architecture                                            |
| `train_model.py`   | Loads dataset, trains the CNN, saves the model                  |
| `prepare_dataset.py` | Creates grouped train/validation/test image splits             |
| `recommend.py`     | Rule-based cosmetics/skincare recommendation engine              |
| `gui_app.py`       | CustomTkinter desktop app — browse a photo, get prediction + tips |
| `app.py`           | Flask browser app with upload, model analysis, and recommendations |
| `web_app.py`       | Optional Streamlit browser prototype                             |
| `inference.py`     | Shared lazy model loading and image prediction                   |
| `requirements.txt` | Python package list                                              |

For instructions on changing layouts, images, and colors, see [guide.md](guide.md).

## Hardware Requirements (as specified)
- Operating System: Windows
- Processor: Intel Core i5 and above
- RAM: 4 GB and above
- Hard Disk: 50 GB

The CNN in `model.py` uses an ImageNet-pretrained MobileNetV2 feature extractor
with a small skin-type classifier head. It accepts 160x160 images and can
train on **CPU**; the pretrained weights are downloaded the first time you
train, so internet access is needed for that initial setup.

## Software Requirements (as specified)
- Python (via **Python IDLE** or run scripts from **Anaconda Prompt**)
- Anaconda (recommended, for easy environment + package management)

---

## Step-by-Step Setup (Windows)

### 1. Install Anaconda
Download and install Anaconda from https://www.anaconda.com/download if you
haven't already.

### 2. Create and activate an environment
Open **Anaconda Prompt** and run:
```bat
conda create -n skinaura python=3.10 -y
conda activate skinaura
```

### 3. Copy the project
Place the entire `SkinAura` folder somewhere convenient, e.g.
`C:\Users\<you>\Desktop\SkinAura`, then in Anaconda Prompt:
```bat
cd C:\Users\<you>\Desktop\SkinAura
```

### 4. Install dependencies
```bat
pip install -r requirements.txt
```
This installs `tensorflow-cpu` (no GPU drivers needed), `optree`, `numpy`, `pillow`,
`matplotlib`, `customtkinter`, `Flask`, and `streamlit` for the desktop and
browser interfaces.

If image analysis reports that Windows Application Control blocked a TensorFlow
or optree native library, the operating system is preventing the model runtime from loading.
Ask your administrator to approve the installed TensorFlow runtime or provide
an approved Python environment. Reinstalling the app cannot override that
policy, and security controls should not be disabled.

### 5. Download the dataset
Get the dataset from Kaggle:
https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset

Extract it and arrange it inside the project folder like this:

```
SkinAura/
  dataset/
    train/
      oily/
      dry/
      normal/
    test/            (optional held-out evaluation split; never used to train)
      oily/
      dry/
      normal/
```
(Folder names are case-insensitive to how they're read, but keep them
consistent — e.g. all lowercase.)

The included archive has 118 dry images in `train/` and 134 dry/normal/oily
images in `test/`. Run this command first to reorganize copies into a usable
dataset without changing the original images:

```bat
python prepare_dataset.py
```

The script groups augmented variants by their original image ID, then creates
`saved_model/prepared_stratified_data/{train,validation,test}/{dry,normal,oily}`.
Each original image group is assigned to only one split. This checkout provides
75 dry, 58 normal, and 40 oily unique source images; split sizes are
stratified per class, not artificially balanced. Counts are printed by the
script. To add more examples, place correctly labelled
images in the matching `train/` or `test/` class folders before preparing.
Preparation refuses to replace an existing output folder; move that folder
aside before rebuilding after you add images.

For a pipeline-only demonstration, `create_demo_dataset.py` can generate
synthetic images and `python train_model.py` can train against them. These
images are not real faces and are not medically valid; replace the generated
`dataset/` with properly licensed real face images before evaluating the
classifier or relying on its predictions.

### 6. Train the model
Still inside `Anaconda Prompt` (with the `skinaura` environment active):
```bat
python train_model.py
```
This will:
- Read only the prepared training split and validate on its separate validation split
- Augment training images with mild geometric, brightness, and color variation
- Train the MobileNetV2 classifier for up to 30 epochs with early stopping
- Evaluate once on the held-out prepared test split
- Save the trained model to `saved_model\skinaura_model.h5`
- Save the class label mapping to `saved_model\class_labels.json`
- Save an accuracy/loss chart to `saved_model\training_history.png`

Training time on an i5 CPU with 4 GB RAM will depend on dataset size —
expect anywhere from a few minutes to under an hour for a few thousand
images at this image resolution and batch size.

### 7. Run the application
```bat
python gui_app.py
```
Or open `gui_app.py` in **Python IDLE** and press **Run > Run Module (F5)**.

For the Flask browser experience, run:
```bat
python app.py
```
Then open `http://127.0.0.1:5000` in your browser. Create a local account or
log in to continue. User accounts and product/routine progress are kept in
`instance\skinaura.sqlite3`; passwords are stored as hashes. The Flask interface
provides profile settings, a dedicated recommendations page, and a daily routine
tracker with editable tasks, per-day check-offs, a streak, and seven-day progress.
Your streak advances only when every active task for the day is complete. The
interface also provides a separate photo-input page and results page, validated
image upload, live camera capture, model confidence, responsive layout, and a
skincare-focused photographic background. Choose
**Use camera** and allow camera access when prompted. Browser camera access
works on `localhost`/`127.0.0.1` and secure HTTPS origins; ordinary remote HTTP
pages cannot request it. Product suggestions are selected using both the skin
profile and the chosen weather condition; the same selection is used in the
downloadable report.

The **AI chat** page answers general skincare questions through GroqCloud. Create
a GroqCloud API key in your Groq account, then open the project's `.env` file
and set `GROQ_API_KEY`:

```dotenv
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b
```

Install the dependencies if needed, then start the app:

```bat
python -m pip install -r requirements.txt
python app.py
```

The default model is `openai/gpt-oss-20b`. To choose another model available to
your Groq account, change `GROQ_MODEL` in `.env`. Groq free-tier
availability and request limits depend on the account and model; SkinAura shows
an actionable message if the key is missing or the service limit is reached.
The `.env` file is excluded from source control. Keep your key private. Chat
messages are sent to GroqCloud for response generation and are not saved by
SkinAura. Responses are educational, not medical diagnoses. Product links shown
with each answer are selected from SkinAura's local recommendation catalog
rather than generated by the AI.

If Keras reports that `optree` is missing, install the project dependencies
using the same Python environment that runs `app.py`:
```bat
python -m pip install -r requirements.txt
```

The prepared set makes all three labels available for training while keeping
source images grouped across splits. It is small and imbalanced, so class
weighting helps but cannot guarantee accuracy. To avoid skin-tone-driven
outputs, SkinAura predicts only dry, normal, or oily; it does not classify
skin tone. The included dataset's tone representation has not been audited,
so add diverse, correctly labelled images for each skin type and check results
across tones before relying on it. Evaluate on the held-out test split. The bundled
`SkinAura_CNN` can still load, but it predates the corrected training workflow
and its original split cannot be verified. The interfaces mark its results as
experimental. Skin type is subtle, and a photo classifier is not a diagnosis.

The older Streamlit interface remains available if needed:
```bat
streamlit run web_app.py
```

In the app, the modern CustomTkinter interface provides a two-panel workflow
with a photo preview, local model status, confidence result, routine starters,
and scrollable skincare notes. Your image stays on the local machine.

The desktop, Flask, and Streamlit interfaces also provide links to example
products, general hydration and balanced-eating reminders, weather-aware
routine advice (based on the condition you select), and a daily routine streak.
In Flask, product checkboxes and routine streaks are stored per local account.
Images are content-validated and processed in memory, not written to disk;
the analysis form requires privacy consent, and downloadable PDF reports omit
the source image. Flask authentication is intended for a single-computer
prototype and is not a substitute for a production hosted identity service.

The photo model predicts only dry, normal, or oily because those are the
classes in its image dataset. Choose Combination or Sensitive (self-assessed)
in the profile selector to see separate product suggestions and morning and
evening task lists. Those two additional profiles are not CNN predictions;
Sensitive is a user-selected guidance profile, not a medical diagnosis.
Analysis results show the ranked image-specific score for each trained class;
the selected class determines the suggested products and routine. The bundled
legacy model can still make incorrect, low-confidence estimates, so retrain it
with `python train_model.py` after preparing the dataset.

In the app:
1. Create a local account or log in.
2. Select a supported face image or capture one, then agree to the image
   processing notice and click **Analyze my skin**.
3. The results appear only after a valid analysis. Review skin-type-specific
   product links and check off products; download the PDF report or mark your
   daily routine complete to build your account's streak.

---

## Notes on the Recommendation Engine
`recommend.py` uses a transparent rule-based lookup (not a second neural
network) to map each predicted class to example product pages and general
routine guidance. Product pages are examples, not endorsements. Hydration,
diet, and weather notes are general wellbeing suggestions, not medical advice.

The app always displays a disclaimer recommending professional
dermatological advice for persistent skin concerns, matching the project's
stated scope as a preliminary/educational tool.

## Extending the Project
- **Model quality**: use a complete, diverse, correctly labelled training set
  and evaluate on a separate representative test set. No architecture can
  compensate for missing or unrepresentative training images.
- **Desktop webcam capture**: add a webcam button in `gui_app.py` using OpenCV
  (`pip install opencv-python`) if camera support is also needed in the desktop interface.
- **More classes**: if you have a dataset with more skin conditions (e.g.
  acne, sensitive), add the folders and update `CLASS_NAMES` in
  `config.py` — the code auto-detects the number of classes from the
  dataset folders.
## Application Screenshots

![SkinAura 1](Screenshot%202026-11-20%20220326.png)

![SkinAura 2](Screenshot%202026-11-20%20220350.png)

![SkinAura 3](Screenshot%202026-11-20%20220431.png)

![SkinAura 4](Screenshot%202026-11-20%20220515.png)

![SkinAura 5](Screenshot%202026-11-20%20220532.png)
