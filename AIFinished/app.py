import tkinter as tk
from tkinter import filedialog, ttk
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk

MODELS = {
    "EMDS7":       ("ProtozoaAlgaeModel_openvino_model/",  "detect"),
    "Microplastic": ("MicroplasticModel_openvino_model/",   "detect"),
    "Bacteria":    ("BacteriaModel_openvino_model/",        "classify"),
}

#Speciile daunatoare din datasetul EMDS7
UNSAFE_EMDS7_SPECIES = {
    "Microcystis",
    "Oscillatoria",
    "Anabaenopsis",
    "Raphidiopsis",
    "Raphidiopsis_2",
    "Phormidium",
    "Merismopedia",
    "Gomphosphaeria",
    "Spirulina",
    "Euglena"
}

#Bacterii daunatoare
UNSAFE_BACTERIA = {
    "E.coli",
    "Escherichia coli",
    "Salmonella",
    "Vibrio cholerae",
    "Legionella",
    "Shigella",
    "Pseudomonas aeruginosa",
    "Enterococcus",
    "Clostridium",
    "Coliform"
}

#EMDS7
CLASS_MAP = {
    "G001": "Oscillatoria",
    "G002": "Ankistrodesmus",
    "G003": "Microcystis",
    "G004": "Gomphonema",
    "G005": "Sphaerocystis",
    "G006": "Cosmarium",
    "G007": "Cocconeis",
    "G008": "Tribonema",
    "G009": "Chlorella",
    "G010": "Tetraedron",
    "G011": "Ankistrodesmus_2",
    "G012": "Brachionus",
    "G013": "Chaenea",
    "G014": "Pediastrum",
    "G015": "Spirogyra",
    "G016": "Coelastrum",
    "G017": "Raphidiopsis",
    "G018": "Spirulina",
    "G019": "Actinastrum",
    "G020": "Scenedesmus",
    "G021": "Staurastrum",
    "G022": "Phormidium",
    "G023": "Fragilaria",
    "G024": "Anabaenopsis",
    "G025": "Coelosphaerium",
    "G026": "Crucigenia",
    "G027": "Achnanthes",
    "G028": "Synedra",
    "G029": "Ceratium",
    "G030": "Pompholyx",
    "G031": "Merismopedia",
    "G032": "Spirogyra_2",
    "G033": "unknown",
    "G034": "Raphidiopsis_2",
    "G035": "Gomphosphaeria",
    "G036": "Euglena",
    "G037": "Euchlanis",
    "G038": "Keratella",
    "G039": "Diversicornis",
    "G040": "Surirella",
    "G041": "Characium",
}

img_path = None
models_loaded = {}

root = tk.Tk()
root.title("Water AI Multi Detector")
root.geometry("1200x900")
root.configure(bg="#0f172a")

style = ttk.Style()
style.theme_use("clam")

header = tk.Frame(root, bg="#1e293b", height=80)
header.pack(fill="x")

tk.Label(
    header,
    text="Water Quality AI Analyzer",
    font=("Segoe UI", 24, "bold"),
    bg="#1e293b",
    fg="white"
).pack(pady=18)

main_frame = tk.Frame(root, bg="#0f172a")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

left_panel = tk.Frame(main_frame, bg="#1e293b", width=260)
left_panel.pack(side="left", fill="y", padx=(0, 15))
left_panel.pack_propagate(False)

right_panel = tk.Frame(main_frame, bg="#1e293b")
right_panel.pack(side="right", fill="both", expand=True)

tk.Label(
    left_panel,
    text="Controls",
    font=("Segoe UI", 18, "bold"),
    bg="#1e293b",
    fg="white"
).pack(pady=15)

emds_var = tk.BooleanVar()
micro_var = tk.BooleanVar()
bacteria_var = tk.BooleanVar()

select_btn = ttk.Button(left_panel, text="Select Image")
select_btn.pack(fill="x", padx=20, pady=10)

ttk.Checkbutton(left_panel, text="EMDS7 (Algae / Protozoa)", variable=emds_var).pack(anchor="w", padx=25)
ttk.Checkbutton(left_panel, text="Microplastic AI", variable=micro_var).pack(anchor="w", padx=25)
ttk.Checkbutton(left_panel, text="Bacteria AI", variable=bacteria_var).pack(anchor="w", padx=25)

load_btn = ttk.Button(left_panel, text="Load Models")
load_btn.pack(fill="x", padx=20, pady=15)

run_btn = ttk.Button(left_panel, text="Run Detection")
run_btn.pack(fill="x", padx=20, pady=(0, 20))


tk.Label(
    left_panel,
    text="Recommended Safety Limits",
    font=("Segoe UI", 13, "bold"),
    bg="#1e293b",
    fg="#facc15"
).pack(pady=(10, 4))

#Limite
recommended_text = """
EMDS7:
• Unsafe if harmful species found

Microplastic:
• Safe: 0–2
• Caution: 3–5
• Unsafe: 6+

Bacteria:
• Unsafe if pathogenic species detected
"""

tk.Label(
    left_panel,
    text=recommended_text,
    font=("Segoe UI", 9),
    bg="#1e293b",
    fg="#cbd5e1",
    justify="left"
).pack(anchor="w", padx=20)

img_label = tk.Label(
    right_panel,
    bg="#334155",
    width=600,
    height=440,
    text="No Image Selected",
    fg="white",
    font=("Segoe UI", 14)
)
img_label.pack(padx=15, pady=10)

verdict_frame = tk.Frame(right_panel, bg="#1e3a2f")
verdict_frame.pack(fill="x", padx=15, pady=(0, 6))

verdict_label = tk.Label(
    verdict_frame,
    text="💧 Water Safety: Run detection to evaluate",
    font=("Segoe UI", 14, "bold"),
    bg="#1e3a2f",
    fg="#94a3b8",
    pady=10
)
verdict_label.pack(fill="x", padx=12)

status_label = tk.Label(
    right_panel,
    text="Status: Waiting for detection",
    font=("Segoe UI", 13, "bold"),
    bg="#1e293b",
    fg="white"
)
status_label.pack(pady=3)

result_text = tk.Text(
    right_panel,
    height=14,
    bg="#0f172a",
    fg="#e2e8f0",
    font=("Consolas", 11),
    wrap="word"
)
result_text.pack(fill="both", expand=True, padx=20, pady=10)

def load_image():
    global img_path
    file = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
    if not file:
        return

    img_path = file
    img = Image.open(file)
    img.thumbnail((600, 440))
    img = ImageTk.PhotoImage(img)

    img_label.config(image=img, text="")
    img_label.image = img

    status_label.config(text="Status: Image Loaded")
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "Image loaded successfully.\n")


def load_models():
    models_loaded.clear()

    if emds_var.get():
        path, task = MODELS["EMDS7"]
        models_loaded["EMDS7"] = YOLO(path, task=task)
    if micro_var.get():
        path, task = MODELS["Microplastic"]
        models_loaded["Microplastic"] = YOLO(path, task=task)
    if bacteria_var.get():
        path, task = MODELS["Bacteria"]
        models_loaded["Bacteria"] = YOLO(path, task=task)

    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f"Loaded models: {', '.join(models_loaded.keys())}\n")

def run_bacteria_classifier(model, detected):
    results = model.predict(source=img_path, verbose=False)
    if not results:
        return None, 0.0

    r = results[0]

    if r.probs is None:
        return None, 0.0

    top1_idx = int(r.probs.top1)
    top1_conf = float(r.probs.top1conf)
    top1_label = r.names.get(top1_idx, str(top1_idx))

    # ONLY keep if confident
    if top1_conf >= 0.70:
        detected.append(f"Bacteria: {top1_label} ({top1_conf:.2f})")
        return top1_label, top1_conf

    #Ignoram nivelul scazut de incredere
    return None, 0.0

def evaluate_water_usage(counts, harmful_detected, harmful_bacteria):
    usage = []

    micro_count = counts.get("Microplastic", 0)

    # ── AGRICULTURE ──
    if harmful_detected or harmful_bacteria:
        usage.append(
            "Agriculture: UNSAFE for vegetables, fruits, grains, or edible crops. "
            "Only possible for ornamental plants after treatment."
        )
    elif micro_count > 5:
        usage.append(
            "Agriculture: LIMITED — suitable only for ornamental plants, flowers, "
            "lawns, or non-edible industrial crops."
        )
    else:
        usage.append(
            "Agriculture: SAFE for vegetables, fruits, gardens, lawns, and most crops."
        )

    # ── ANIMALS ──
    if harmful_detected or harmful_bacteria:
        usage.append(
            "Animals: UNSAFE for dogs, cats, cattle, horses, goats, pigs, poultry, or rabbits."
        )
    elif micro_count > 3:
        usage.append(
            "Animals: LIMITED — may be used cautiously for large livestock "
            "(cattle, horses) but NOT recommended for pets, poultry, or small animals."
        )
    else:
        usage.append(
            "Animals: SAFE for dogs, cats, cattle, horses, goats, pigs, chickens, and rabbits."
        )

    # ── HOUSEHOLD ──
    if harmful_detected or harmful_bacteria:
        usage.append(
            "Household: UNSAFE for bathing, cleaning food, washing dishes, or recreation."
        )
    elif micro_count > 2:
        usage.append(
            "Household: LIMITED — usable for toilet flushing, floor cleaning, "
            "vehicle washing, or irrigation only. Avoid bathing or cooking."
        )
    else:
        usage.append(
            "Household: SAFE for washing, bathing, cleaning, and general non-drinking purposes."
        )

    # ── AQUACULTURE ──
    if harmful_detected:
        usage.append(
            "Aquaculture: UNSAFE for fish, shrimp, or aquatic farming."
        )
    elif harmful_bacteria:
        usage.append(
            "Aquaculture: LIMITED — treatment required before use for fish farming."
        )
    else:
        usage.append(
            "Aquaculture: SAFE for most fish ponds and aquatic livestock."
        )

    # ── INDUSTRIAL USE ──
    if harmful_detected or harmful_bacteria:
        usage.append(
            "Industrial: LIMITED — only for non-contact cooling or mechanical cleaning after filtration."
        )
    elif micro_count > 5:
        usage.append(
            "Industrial: SAFE for basic machinery washing, cooling systems, and construction."
        )
    else:
        usage.append(
            "Industrial: SAFE for most industrial, cleaning, and construction applications."
        )

    return usage


def run_detection():
    if img_path is None or not models_loaded:
        return

    img = cv2.imread(img_path)
    output = img.copy()

    detected = []
    counts = {}
    harmful_detected = False
    bacteria_label = None
    bacteria_conf = 0.0

    for model_name, model in models_loaded.items():

        if model_name == "Bacteria":
            bacteria_label, bacteria_conf = run_bacteria_classifier(model, detected)
            if bacteria_label:
                detected.append(f"Bacteria: {bacteria_label} ({bacteria_conf:.2f})")
            continue

        results = model.predict(source=img_path, conf=0.25, verbose=False)

        if not results:
            counts[model_name] = 0
            continue

        r = results[0]
        model_count = 0

        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.50:
                continue

            cls_id = int(box.cls[0])
            raw = r.names.get(cls_id, str(cls_id))

            if model_name == "EMDS7":
                label = CLASS_MAP.get(raw, raw)
                if "unknown" in label.lower():
                    continue
                if label in UNSAFE_EMDS7_SPECIES:
                    harmful_detected = True
            else:
                label = raw

            detected.append(f"{model_name}: {label} ({conf:.2f})")
            model_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0,255,0) if model_name=="EMDS7" else (255,80,80)

            cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output, label, (x1, max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        counts[model_name] = model_count

    harmful_bacteria = bacteria_label in UNSAFE_BACTERIA if bacteria_label else False

    unsafe_reasons = []

    if harmful_detected:
        unsafe_reasons.append("harmful algae detected")
    if harmful_bacteria:
        unsafe_reasons.append(f"harmful bacteria: {bacteria_label}")
    if counts.get("Microplastic", 0) >= 6:
        unsafe_reasons.append("high microplastic contamination")

    # Verdict
    if unsafe_reasons:
        verdict_frame.config(bg="#450a0a")
        verdict_label.config(
            text="⛔ Water UNSAFE — " + " | ".join(unsafe_reasons),
            bg="#450a0a",
            fg="#fca5a5"
        )
    else:
        verdict_frame.config(bg="#14532d")
        verdict_label.config(
            text="✅ Water SAFE",
            bg="#14532d",
            fg="#86efac"
        )

    # Show image
    img_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    img_out = Image.fromarray(img_out)
    img_out.thumbnail((600,440))
    img_out = ImageTk.PhotoImage(img_out)

    img_label.config(image=img_out, text="")
    img_label.image = img_out

    # Results
    result_text.delete("1.0", tk.END)

    if detected:
        for d in detected:
            result_text.insert(tk.END, f"• {d}\n")
    else:
        result_text.insert(tk.END, "No detections found.\n")

    result_text.insert(tk.END, "\n── Water Usage Recommendations ──\n")

    usage = evaluate_water_usage(
        counts,
        harmful_detected,
        harmful_bacteria
    )

    for item in usage:
        result_text.insert(tk.END, f"• {item}\n")

select_btn.config(command=load_image)
load_btn.config(command=load_models)
run_btn.config(command=run_detection)

root.mainloop()
