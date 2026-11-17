"""
ClaimLens — Export Trained Models for TF Serving
Run this in your Google Colab AFTER training is complete.

This converts your Keras .h5 or in-memory model into the
SavedModel format that TensorFlow Serving requires.

CHANGE: paths to point to your actual trained model files.
"""
import tensorflow as tf
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — Load your trained Keras model
# CHANGE: this path to wherever you saved your model in Colab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Damage model (EfficientNetB3 fine-tuned on CarDD/VeHIDE) ──────────────
DAMAGE_MODEL_PATH = "/content/drive/MyDrive/vehide_efficientnetb3_best.h5"
# CHANGE: update this path to your saved .h5 file

damage_model = tf.keras.models.load_model(DAMAGE_MODEL_PATH)
print("Damage model loaded. Summary:")
damage_model.summary()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — Export to SavedModel format
# The "1" subfolder is the version number — required by TF Serving
# CHANGE: export path if your Drive folder is different
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPORT_BASE = "/content/drive/MyDrive/claimlens_models"

# Export damage model
damage_export_path = os.path.join(EXPORT_BASE, "damage_model", "1")
tf.saved_model.save(damage_model, damage_export_path)
print(f"✅ Damage model exported to: {damage_export_path}")

# ── Verify the export ─────────────────────────────────────────────────────
loaded = tf.saved_model.load(damage_export_path)
infer  = loaded.signatures["serving_default"]
print("Export signature:", infer.structured_input_signature)

# ── Test with a dummy input ───────────────────────────────────────────────
import numpy as np
dummy = tf.constant(np.random.rand(1, 300, 300, 3), dtype=tf.float32)
result = infer(keras_tensor=dummy)   # CHANGE key if your input name differs
print("Test prediction shape:", list(result.values())[0].shape)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — Repeat for vehicle model when trained
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# VEHICLE_MODEL_PATH = "/content/drive/MyDrive/vehicle_model_best.h5"
# vehicle_model = tf.keras.models.load_model(VEHICLE_MODEL_PATH)
# vehicle_export = os.path.join(EXPORT_BASE, "vehicle_model", "1")
# tf.saved_model.save(vehicle_model, vehicle_export)
# print(f"✅ Vehicle model exported to: {vehicle_export}")

print("\n✅ All models exported. Copy the claimlens_models folder")
print("   to your server and mount it into the TF Serving container.")
print("\n   Folder structure should be:")
print("   claimlens_models/")
print("     damage_model/")
print("       1/")
print("         saved_model.pb")
print("         variables/")
print("     vehicle_model/")
print("       1/  ...")
