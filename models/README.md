# Model weights

Place the trained YOLO weights here as **`best.pt`**:

```
models/best.pt
```

Or keep them anywhere and point to them with an environment variable:

```powershell
# Windows PowerShell
$env:PLATE_MODEL_PATH = "D:\path\to\best.pt"
```

```bash
# Linux / macOS
export PLATE_MODEL_PATH=/path/to/best.pt
```

The model was trained on 4 classes: `License_Plate`, `cars`, `motorcyle`, `truck`
(class names exactly as in the training dataset).
