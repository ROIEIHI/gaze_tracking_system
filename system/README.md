# Gaze Tracking Reading Analysis System (V4.1)

Webcam-based gaze tracking for Hebrew reading experiments. Each session trains a personal gaze model for the participant, then shows the texts and questions of the experiment script and records which word they look at, fixation by fixation.

- Runs on Windows with an ordinary webcam. No eye-tracking hardware is needed.
- Hebrew, right-to-left text only.
- Everything you normally change lives in [`config.py`](config.py).

For installation, see [`SETUP_README.md`](SETUP_README.md). For how the code is organized, see [`CODE_MODULES_EXPLANATION.md`](CODE_MODULES_EXPLANATION.md).

## Quick start

From the repository folder, in PowerShell:

```powershell
.\.venv\Scripts\python.exe system\start.py
```

Use the `.venv` Python explicitly. A plain `python` may start a different installation that lacks the packages.

## Running a session

1. **Participant details.** Fill in first name and surname (English letters only), a 9-digit ID number, the group (R regular or G flexible) and the ADHD check box, then press Start. Choose whether to go fullscreen.
   Before anything starts, the program checks that `config.py` defines every setting, that the texts fit on screen, and that the camera opens. Any problem stops it here with a message.
2. **Position.** A live camera view shows a box. The participant sits so the face is inside it. Once the face has been in place for a second, a green "FACE STABLE - Ready for calibration" message appears and the tester presses SPACE.
3. **Calibration.** Press SPACE to start. The participant clicks the centre point for the baseline, then clicks each red point (37 in total) and keeps looking at it while it is recorded. ESC cancels at any step.
4. **Training.** The model trains for about a minute while a "please keep still" camera view stays on screen.
5. **Accuracy check.** The gaze error is shown in degrees. If it is over `ERROR_THRESHOLD_DEGREES` and `ERROR_THRESHOLD_ENABLED` is on, a dialog offers to recalibrate with a fresh dataset.
6. **Reading.** The opening screen shows the instructions, with a small camera view. The tester presses D to start, which only works once the face is inside the box. The texts and questions then run in the order of the experiment script.
7. **End.** The closing message appears after the last text. ESC ends the session and saves everything.

### During reading

| Screen | What happens |
|---|---|
| Text page | D goes to the next page, A to the previous one |
| Yes/No question | The participant clicks YES or NO. D does nothing here |
| 1–5 rating | The participant shows the answer with their fingers to the camera. The tester notes it and presses D |
| Position pause | Appears if the face is out of the box for 2 s on a text page. Reading resumes by itself on the same page 1 s after the face is back |
| Closing message | ESC ends the session |

E saves the data at any point without ending the session. The keyboard must be on the English layout: on the Hebrew layout A, D and E do nothing.

Gaze is only recorded on text pages, and only while the face is inside the box. Short drifts out of the box (under 2 s) leave a gap in the data without pausing.

## Changing the experiment

[`config.py`](config.py) opens with a numbered index. Search for the number, for example `# 3.`, to jump to a section.

| Section | Settings you are most likely to change |
|---|---|
| 1. Session options | `SHOW_GAZE_POINT`, `VIDEO_RECORDING_ENABLED`, `ERROR_THRESHOLD_ENABLED`, `ERROR_THRESHOLD_DEGREES` |
| 2. Lab setup | `SCREEN_DIAGONAL_CM`, `VIEWING_DISTANCE_CM`, `CAMERA_INDEX` |
| 3. Text display | `TEXT_ROWS_PER_PAGE`, `TEXT_FONT_SIZE`, `TEXT_LINE_SPACING`, `TEXT_MARGIN_X`, `TEXT_MARGIN_Y`, `INSTRUCTION_FONT_SIZE` |
| 4. Gaze analysis | `FIXATION_THRESHOLD`, `SACCADE_VELOCITY_THRESHOLD`, `WORD_PROXIMITY_THRESHOLD` |
| 5. Participant position | The face box, `MIN_FACE_SIZE`, and the reading pause: `OOB_CHECK_ENABLED`, `OOB_PAUSE_AFTER_SECONDS`, `OOB_RESUME_AFTER_SECONDS` |
| 13. Experiment script | `OPENING_TITLE`, `OPENING_INSTRUCTIONS`, `REPOSITION_MESSAGE`, `EXPERIMENT`, `CLOSING_MESSAGE` |

`TEXT_MARGIN_Y = 0.22` centres 4 rows at font 40 and spacing 120. If you change the rows, font or spacing, adjust it by hand.

`MIN_FACE_SIZE` applies at setup and during reading. A face narrower than this, in camera pixels, counts as out of position.

### The experiment script

`EXPERIMENT` lists the texts in the order they are shown. Each text has a number, a type that is saved with the data but never shown, and its steps from top to bottom:

```python
{"number": 1, "type": "קל", "steps": [
    ("text", """first part of the text..."""),
    ("probe", UNDERSTOOD_PROBE),
    ("text", """second part of the text..."""),
    ("probe", UNDERSTOOD_PROBE),
    ("rating", UNDERSTANDING_RATING),
    ("probe", "comprehension question?", "Yes"),
]},
```

| Step | Meaning |
|---|---|
| `("text", content)` | Part of the text. It is split into as many pages as needed. Only these steps record gaze |
| `("probe", question)` | A Yes/No question, answered by clicking |
| `("probe", question, "Yes"/"No")` | The same, with the correct answer, so the response is scored |
| `("rating", question)` | A question with no buttons. The tester records the answer and presses D |

A text that cannot fit on the screen, an unknown step type or a missing setting stops the program before calibration, with a message naming the problem.

If your editor says `config.py` changed on disk, reload it before you type. Saving an older copy over a newer one removes the newer settings.

## Output files

Each session gets its own folder in `system/user_data`, named `First_Surname_ID_Group_ADHD|NoADHD_YYYYMMDD-HHMMSS`:

```
<session folder>/
├── calibration/calibration_data_<time>.csv     Calibration samples, the model's training data
├── models/                                     gaze_model.pkl, scaler.pkl, model_info.txt
├── accuracy_check.json                         Gaze error of every calibration attempt
├── session_summary.txt                         Overview: error, pass/fail, position pauses
└── analysis/session_<time>/
    ├── analysis/
    │   ├── eye_movement_data_<time>.csv        One row per fixation
    │   ├── probe_answers_<time>.csv            One row per question
    │   └── session_events_<time>.csv           One row per stretch out of the box
    └── recordings/
        ├── camera_feed_<...>.mp4               Webcam video
        ├── overlay_display_<...>.mp4           Screen video with the gaze markers
        └── video_metadata_<time>.json
```

The CSV files are UTF-8 with a BOM, so Excel shows the Hebrew correctly.

### `eye_movement_data_<time>.csv`

| Column | Meaning |
|---|---|
| `Fixation_Order` | Order of the fixation in the session |
| `Text_Number` | Text number from the experiment script |
| `Text_Type` | The text's type, for example קל or קשה |
| `Part_Number` | Part of the text (1 or 2) |
| `Page_Number` | Page on screen, counted through the whole session |
| `Line` | Line within the text, counted through all its parts and pages |
| `Word_Index` | Position of the word in the text, counted the same way. Blank when the gaze was not on a word |
| `Fixated_Word` | The word, or `Unknown` |
| `Fixation_X_Screen`, `Fixation_Y_Screen` | Mean gaze position, in pixels of the 1080×720 screen image |
| `Fixation_Duration` | Milliseconds |
| `Time_from_Stimulus_Onset` | Milliseconds since the first text page appeared |
| `Pupil_Size` | A relative index from the iris diameter in camera pixels (2 + diameter / 50 × 3, clamped to 2–6). It is not a size in millimetres |
| `Blink_Frequency` | Blinks per minute over the last 60 s |
| `Probe_Answer` | The answer to the "did you understand" question after this part |

`Text_Number` together with `Word_Index` identifies exactly one word of the experiment text.

### `probe_answers_<time>.csv`

One row per question, in order: `Question_Order`, `Text_Number`, `Text_Type`, `Part_Number`, `Question_Type` (`probe` or `rating`), `Question`, `Answer`, `Correct_Answer` and `Is_Correct`. Unanswered questions show `N/A`. Ratings are left blank for the tester to fill in. Answers are saved even when a page has no gaze data.

### `session_events_<time>.csv`

One row per stretch with the face out of the box, once reading has started: `Event_Order`, `Start_ms` (since the first text page), `Duration_ms`, `Screen` (`text`, `probe`, `rating` or `message`), `Text_Number`, `Part_Number`, `Page_Number`, `Reason` and `Paused` (whether reading paused). The session summary shows the total.

## How it works

1. MediaPipe Face Mesh finds the face and iris landmarks in each camera frame.
2. Nine features are measured: the iris position within the face for each eye, head pitch and yaw relative to a baseline, and the head translation from OpenCV `solvePnP`. Two interaction features are added, for 11 in total.
3. A multi-output XGBoost model, tuned by a grid search for each participant, maps the features to a point on the 1080×720 screen image.
4. During reading, the gaze point is smoothed (`smoothing_factor = 0.3`). Fast movements over `SACCADE_VELOCITY_THRESHOLD` count as saccades and are not recorded.
5. Each sample is matched to the nearest word centre within `WORD_PROXIMITY_THRESHOLD` pixels. Consecutive samples on the same word form a fixation.

The accuracy check uses 20% of the calibration samples, held back from training. Those samples come from the same calibration points the model trained on, so the error it reports is lower than the error at new screen positions.

## Other tools

| Script | Use |
|---|---|
| `run_debug_cursor.py` | Runs the reading screens with the mouse as the gaze, with no camera or model. Useful for checking texts and word matching. D also moves past questions, and the data goes to `system/user_data/session_<time>` |
| `main.py` | An older command-line flow without the participant form or the accuracy check. Not for experiment sessions |
| `setup.py` | Installs the requirements into the Python that runs it |

## Project structure

```
system/
├── start.py                  Launcher: checks the packages, then starts the GUI
├── gui.py                    Participant form and the session workflow
├── calibration.py            Face position, calibration points and training preview
├── model_training.py         Feature engineering and the XGBoost model
├── prediction.py             Reading loop, gaze recording and data export
├── text_layout.py            Experiment sequence, Hebrew text layout and drawing
├── eye_movement_analyzer.py  Gaze velocity for the saccade filter
├── run_debug_cursor.py       Mouse-driven test of the reading screens
├── main.py                   Older command-line flow
├── setup.py                  Installs the requirements
├── config.py                 All settings and the experiment script
└── requirements.txt          Python packages
```

## Known limitations

- Windows only. The Hebrew fonts are loaded from `C:\Windows\Fonts`.
- The text layout is right-to-left only.
- A, D and E need the English keyboard layout.
- `Pupil_Size` is a relative index, not a physical pupil measurement.
