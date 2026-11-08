# Code Modules

How the code is organized, for anyone changing or maintaining it. For running sessions see [`README.md`](README.md).

## Session flow

```
start.py ──> gui.py  (participant form, startup checks)
              │
              ├─ calibration.py     position check, 37 calibration points ──> calibration CSV
              ├─ model_training.py  outliers, features, XGBoost grid search ──> gaze_model.pkl, scaler.pkl
              ├─ gui.py             accuracy check, optional recalibration ──> accuracy_check.json
              └─ prediction.py      reading loop ──> fixations, answers, position events, videos
                   ├─ text_layout.py           experiment sequence, page layout and drawing
                   └─ eye_movement_analyzer.py gaze velocity for the saccade filter
```

Every module reads its settings with `from config import *`.

## Modules

### `config.py`
All settings and the experiment script, in 13 numbered sections with an index at the top. The paths in section 10 are built from the file's own location, so data always goes to `system/user_data` whatever folder the program is started from. Folders are created only when something is written to them.

### `start.py`
Checks that the main packages import, then runs `gui.py` with the same Python.

### `gui.py`
- `GazeTrackingGUI` holds the participant form and runs the workflow on a background thread: calibration, training, accuracy check, reading, then the session summary.
- `validate_inputs` runs before anything starts. It checks the form, calls `missing_config_settings`, builds the reading sequence (which raises if text cannot fit), and checks that the camera opens.
- `missing_config_settings` parses every module that does `from config import *` and lists the upper-case names it reads that `config.py` does not define.
- `accuracy_accepted` converts the model's test error to degrees with `calibration.pixels_to_degrees`, offers recalibration when it is over the threshold, and appends every attempt to `accuracy_check.json`.

### `calibration.py`
- `face_position_problem(landmarks, frame_shape)` is the single rule for whether a face is usable: big enough (`MIN_FACE_SIZE`) and centred (the face box plus `FACE_BOUNDARY_SENSITIVITY`). It returns a message, or `""` when the face is fine. Setup and reading both use it.
- `draw_face_box` draws the box, and `pixels_to_degrees` converts an error to degrees of visual angle.
- `GazeCalibrator` runs the setup preview, the baseline, and the calibration loop: 21 fixed points plus a 4×4 grid, 11 frames each. It saves the calibration CSV.
- `training_preview()` is a context manager that shows a camera view on a thread while the model trains.

### `model_training.py`
The core model. Changing it changes the results, so check before editing.

`GazeModelTrainer` loads the calibration CSV, removes outliers (IQR), adds two interaction features to the nine measured ones, and holds out 20% of the rows for testing. It adds 5 px of noise to the training targets, then tunes a multi-output `XGBRegressor` with `GridSearchCV` (243 combinations × 3 folds). It saves the model, the scaler and `model_info.txt`, and leaves the metrics in `training_results`.

### `text_layout.py`
- `build_reading_sequence()` turns `OPENING_*`, `EXPERIMENT` and `CLOSING_MESSAGE` into the list of screens a session steps through (see below). It raises `ValueError` for text that cannot fit or an unknown step type.
- `paginate_text` wraps a text by measured pixel width into rows and pages. `render_page` draws a page right to left and returns the position of every word exactly where it was drawn, with its line and word number.
- `layout_instructions` and `draw_instructions` wrap and centre the opening instructions.

### `prediction.py`
- `VideoRecorder` writes the camera and screen videos on a background thread. Frames are dropped rather than slowing the reading loop.
- `TextReadingGazePredictor.run_text_reading_analysis()` is the reading loop. Each frame it reads the camera, predicts the gaze, notes the screen being shown, updates the position check, draws the screen, records gaze if allowed, and handles keys.
- `_update_position` pauses reading after `OOB_PAUSE_AFTER_SECONDS` out of the box on a text page, resumes after `OOB_RESUME_AFTER_SECONDS` back inside, and logs each stretch out of the box.
- `_add_gaze_point_to_analysis` drops samples faster than `SACCADE_VELOCITY_THRESHOLD`, matches the rest to the nearest word, and stores them.
- `_export_fixation_data` saves the answers file and the position events first, then groups consecutive samples on the same word position into fixations.

### `eye_movement_analyzer.py`
Gives the reading loop the current gaze velocity that the saccade filter uses. It also contains older classes and metrics that the current export does not use.

### `run_debug_cursor.py`
`DebugGazePredictor` replaces the camera and model with the mouse position, for testing texts, layout and word matching. It has no position check, and its clicks do not answer questions.

### `main.py` and `setup.py`
`main.py` is an older command-line flow that saves to `system/calibration_data` and `system/models`. It has no participant form or accuracy check. `setup.py` installs `requirements.txt` into the Python that runs it.

## Reading sequence

`build_reading_sequence()` returns one dictionary per screen:

| `type` | Comes from | Notes |
|---|---|---|
| `message` | Opening and closing screens | The opening screen has `details` (the instructions) and `check_position: True` |
| `page` | A `text` step | One per page on screen. Has `text`, `part`, `page_number`, `content` (rows of words), `first_line` and `first_word` |
| `probe` | A `probe` step | Has `correct_answer` when the script gives one. Moved on by a click only |
| `rating` | A `rating` step | Moved on by D |

## Rules that keep the data clean

- Gaze is recorded only on `page` screens, and only while the face is in the box when `OOB_CHECK_ENABLED` is on.
- The screen on display is read before drawing, because a click on a question moves to the next step while that frame is drawn.
- The reading clock (`Time_from_Stimulus_Onset`, `Start_ms`) starts when the first text page appears.
- Answers are saved from the sequence, not from fixation rows, so a page with no gaze cannot lose an answer.
- Fixations group by word position (`Text_Number` + `Word_Index`), not by spelling, so two copies of the same word stay separate.
