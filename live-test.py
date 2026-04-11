import gradio as gr
import cv2
import time
from pipeline_mnet import PoseExtractor, FeatureEngineer
from classifiers.rule_based import RuleBasedClassifier
from classifiers.cnn_classifier import CNNClassifier
from classifiers.mobilenet_classifier import PretrainedImageClassifier

# --- INITIALIZE PIPELINE ---
extractor = PoseExtractor()
engineer = FeatureEngineer(history=35)
rule_clf = RuleBasedClassifier()

# Optional: Load heavy models only if files exist
try:
    cnn_clf = CNNClassifier()
    mnet_clf = PretrainedImageClassifier(model_path="models/mobilenet_gesture.pt")
except Exception as e:
    print(f"Deep learning models skipped: {e}")
    cnn_clf, mnet_clf = None, None

def live_predict(frame):
    if frame is None:
        return None, "No feed"

    # Gradio sends RGB, OpenCV usually wants BGR, but your pipeline 
    # likely uses RGB if using MediaPipe/MobileNet. 
    # Just in case, we flip it horizontally for the 'mirror' effect.
    frame = cv2.flip(frame, 1)
    
    t = time.perf_counter()
    pose_frame = extractor.extract(frame, t)

    if not pose_frame:
        return frame, "No person detected"

    fv = engineer.update(pose_frame)
    
    # Get Predictions
    results = []
    
    # 1. Rule Based
    res_rb = rule_clf(fv)
    results.append(f"Rule-Based: {res_rb}")
    
    # 2. CNN (if loaded)
    if cnn_clf:
        res_cnn = cnn_clf(fv, raw_landmarks=pose_frame.landmarks)
        results.append(f"CNN: {res_cnn}")
        
    # 3. MobileNet (if loaded)
    if mnet_clf:
        res_mnet = mnet_clf(frame)
        results.append(f"MobileNet: {res_mnet}")

    # Overlay results on the image for the UI
    display_frame = frame.copy()
    for i, text in enumerate(results):
        cv2.putText(display_frame, text, (10, 30 + (i * 30)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return display_frame, "\n".join(results)

# --- GRADIO LAYOUT ---
with gr.Blocks(title="GestureBall Live") as demo:
    gr.Markdown("# 🏀 GestureBall Live Inference")
    gr.Markdown("Real-time gesture detection using Rule-based, CNN, and MobileNet models.")
    
    with gr.Row():
        input_cam = gr.Image(sources=["webcam"], streaming=True, label="Live Feed")
        output_img = gr.Image(label="Model Visualization")
    
    with gr.Row():
        output_text = gr.Textbox(label="Detection Results")

    # This creates the loop: webcam -> live_predict -> [output_img, output_text]
    input_cam.stream(fn=live_predict, inputs=input_cam, outputs=[output_img, output_text])

if __name__ == "__main__":
    demo.launch()