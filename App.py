from pathlib import Path
import shutil
import zipfile

source = Path("/mnt/data/App_fixed.py")
app = Path("/mnt/data/App.py")
requirements = Path("/mnt/data/requirements.txt")
readme = Path("/mnt/data/README.md")
zip_path = Path("/mnt/data/diabetes-risk-prediction-deployment.zip")

# Copy the clean Streamlit application.
shutil.copyfile(source, app)

requirements.write_text(
    "\n".join(
        [
            "streamlit==1.60.0",
            "pandas==3.0.3",
            "numpy==2.5.1",
            "scikit-learn==1.9.0",
            "joblib==1.5.2",
            "xgboost==3.2.0",
            "",
        ]
    ),
    encoding="utf-8",
)

readme.write_text(
    """# Diabetes Risk Prediction

Streamlit application developed by the Materials Modelling & Simulation Laboratory,
Department of Physics, Allama Iqbal Open University, Islamabad, Pakistan.

## Required repository files

- `App.py`
- `requirements.txt`
- `diabetes_xgboost_model.pkl`
- `diabetes_scaler.pkl`
- `logo.png` (optional)

## Deployment

1. Replace the existing GitHub `App.py` completely with the supplied clean `App.py`.
2. Replace the existing `requirements.txt`.
3. Confirm that the model and scaler files are in the repository root.
4. In Streamlit Community Cloud, deploy `App.py` from the `main` branch.
5. Reboot the application after committing the changes.

Do not place file-generation statements such as `app_code`, `write_text`,
or `/mnt/data` paths inside `App.py`.
""",
    encoding="utf-8",
)

# Validate the generated application.
app_text = app.read_text(encoding="utf-8")
compile(app_text, str(app), "exec")

for forbidden in (
    "app_code =",
    "write_text(",
    'Path("/mnt/data',
    "BASE_FOLDER",
):
    if forbidden in app_text:
        raise RuntimeError(f"Forbidden generator code remains: {forbidden}")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(app, arcname="App.py")
    archive.write(requirements, arcname="requirements.txt")
    archive.write(readme, arcname="README.md")

print("Created and validated:")
print(app)
print(requirements)
print(readme)
print(zip_path)
print(f"App.py lines: {len(app_text.splitlines())}")
