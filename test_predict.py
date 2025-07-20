from inference_core import predict_region_from_path

# === Replace this with the path to your test image ===
image_path = "test_image.jpg"

# === Run prediction ===
label, confidence, top3 = predict_region_from_path(image_path)

print(f"\n🧠 Predicted region: {label}")
print(f"🔒 Confidence: {confidence:.2%}")
print("🎯 Top 3 predictions:")
for cls, prob in top3:
    print(f"  - {cls}: {prob:.2%}")
