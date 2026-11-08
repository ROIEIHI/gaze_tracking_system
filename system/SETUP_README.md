# Setup Guide

How to install the system on a Windows lab computer and check that it works.

## Requirements

| Item | Requirement |
|---|---|
| Operating system | Windows 10 or 11. Hebrew fonts are loaded from `C:\Windows\Fonts` |
| Python | 3.11, the tested version (3.11.9) |
| Camera | A webcam at the top centre of the screen |
| Screen | Its diagonal and the viewing distance go into `config.py` |

## Install

Open PowerShell in the repository folder (the one that contains `system`), then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r system\requirements.txt
```

`system\setup.py` does the same install and creates `system\user_data`. It installs into whichever Python runs it, so run it with the `.venv` Python:

```powershell
.\.venv\Scripts\python.exe system\setup.py
```

### Check the install

```powershell
.\.venv\Scripts\python.exe -c "import cv2, mediapipe, numpy, pandas, sklearn, xgboost, PIL, bidi.algorithm, arabic_reshaper, tkinter; print('Installation OK')"
```

### Tested versions

The system was last tested with these versions. `requirements.txt` sets minimum versions and pins MediaPipe.

| Package | Version |
|---|---|
| opencv-python | 5.0.0.93 |
| mediapipe | 0.10.14 |
| numpy | 2.4.6 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| xgboost | 3.2.0 |
| scipy | 1.17.1 |
| joblib | 1.6.0 |
| Pillow | 13.0.0 |
| python-bidi | 0.6.12 |
| arabic-reshaper | 3.0.1 |

## Run

```powershell
.\.venv\Scripts\python.exe system\start.py
```

`start.py` checks the main packages and then starts the GUI with the same Python. Always start it with the `.venv` Python: other Python installations on the computer may lack the packages or have different versions.

## Lab setup

Set these in `config.py` to match the room before running participants:

| Setting | Meaning |
|---|---|
| `SCREEN_DIAGONAL_CM` | Physical screen diagonal. A 14-inch screen is 35.56 cm |
| `VIEWING_DISTANCE_CM` | Distance from the eyes to the screen |
| `CAMERA_INDEX` | Which camera to use, 0 for the first one |
| `MIN_FACE_SIZE` | The narrowest face accepted, in camera pixels. A face narrower than this counts as too far from the camera |

These two settings convert the gaze error to degrees, so wrong values give a wrong accuracy check.

- Light the face evenly from the front, and avoid a bright window behind the participant.
- Close other programs that use the camera before starting.
- Switch the keyboard to the English layout. On the Hebrew layout A, D and E do nothing.

## Troubleshooting

| Problem | What to do |
|---|---|
| "Configuration Error" at Start | `config.py` is missing the settings listed, or they are misspelled. Restore them, for example with `git diff system/config.py` |
| "Text Layout Error" at Start | A text, word or the opening instructions do not fit on screen. Shorten the text, or reduce the font or rows in section 3 of `config.py` |
| "Camera Error" at Start | Another program is using the camera, or `CAMERA_INDEX` is wrong |
| "FACE STABLE" never appears at setup | The face is too far (`MIN_FACE_SIZE`) or off-centre, as the red message says. Move the participant, or adjust the face box in section 5 |
| A, D and E do nothing | Switch the keyboard to the English layout |
| Reading keeps pausing | The face keeps leaving the box. Re-seat the participant. `OOB_PAUSE_AFTER_SECONDS` sets the delay, and `OOB_CHECK_ENABLED = False` turns the pause off |
| Settings disappear from `config.py` | An editor saved an older copy over a newer one. Reload the file when the editor says it changed on disk |
| `DLL load failed` when starting | Windows Smart App Control can block a library now and then. Start again, and check Windows Security if it repeats |
| The program closes with code 3221225477 | A crash inside a library. Run `$env:PYTHONFAULTHANDLER = "1"` in the same PowerShell window first, then start again. The next crash then prints the failing line |
| Hebrew is garbled in Excel | Only files saved before V4.1 have this. Open them with Data > From Text/CSV and choose UTF-8 |

## Updating

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r system\requirements.txt
```

Your local changes to `config.py` stay in place. If `git pull` reports a conflict in that file, keep your values and take the new settings.
