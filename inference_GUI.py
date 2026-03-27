from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from inference_demo import (
    TARGET_SAMPLE_RATE,
    get_class_names,
    infer_input_shape,
    load_inference_model,
    predict_file,
    record_wav,
)


DEFAULT_MODEL_PATH = "outputs/transfer_runs/run_001/transfer_mobilenetv2_best.keras"
DEFAULT_DATASET_PATH = r"C:\Users\Alex\OneDrive\Desktop\Year 3\0 - Dissertation\Datasets\Google Speech Commands"
DEFAULT_DURATION = "2.0"
DEFAULT_NUM_CLASSES = 30
TOP_K = 5


class InferenceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Speech Command Inference")
        self.root.geometry("900x650")
        self.root.minsize(820, 560)

        self.model = None
        self.class_names: list[str] | None = None
        self.model_loaded = False

        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self.dataset_path_var = tk.StringVar(value=DEFAULT_DATASET_PATH)
        self.duration_var = tk.StringVar(value=DEFAULT_DURATION)
        self.status_var = tk.StringVar(value="Load a model to begin.")
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.input_var = tk.StringVar(value="Input: -")

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        settings = ttk.LabelFrame(main, text="Configuration", padding=10)
        settings.grid(row=0, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Model path").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(settings, textvariable=self.model_path_var).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(settings, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=(8, 0), pady=6)

        ttk.Label(settings, text="Dataset path").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(settings, textvariable=self.dataset_path_var).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(settings, text="Browse", command=self.browse_dataset).grid(row=1, column=2, padx=(8, 0), pady=6)

        ttk.Label(settings, text="Mic duration (s)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(settings, textvariable=self.duration_var, width=12).grid(row=2, column=1, sticky="w", pady=6)
        ttk.Button(settings, text="Load model", command=self.load_model_async).grid(row=2, column=2, padx=(8, 0), pady=6)

        actions = ttk.LabelFrame(main, text="Actions", padding=10)
        actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)

        ttk.Button(actions, text="Predict from WAV file", command=self.predict_wav_async).grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=4)
        ttk.Button(actions, text="Record microphone and predict", command=self.predict_mic_async).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(actions, text="Clear results", command=self.clear_results).grid(row=0, column=2, sticky="ew", padx=(8, 0), pady=4)

        results = ttk.LabelFrame(main, text="Results", padding=10)
        results.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        results.columnconfigure(0, weight=1)
        results.rowconfigure(3, weight=1)

        ttk.Label(results, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(results, textvariable=self.input_var, font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(results, textvariable=self.prediction_var, font=("Segoe UI", 14, "bold")).grid(row=2, column=0, sticky="w", pady=(4, 10))

        columns = ("rank", "label", "probability")
        self.tree = ttk.Treeview(results, columns=columns, show="headings", height=10)
        self.tree.heading("rank", text="#")
        self.tree.heading("label", text="Label")
        self.tree.heading("probability", text="Probability")
        self.tree.column("rank", width=60, anchor="center")
        self.tree.column("label", width=220, anchor="w")
        self.tree.column("probability", width=140, anchor="center")
        self.tree.grid(row=3, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(results, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=3, column=1, sticky="ns")

        log_frame = ttk.LabelFrame(main, text="Log", padding=10)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        self.log_text = tk.Text(log_frame, height=10, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.grid(row=0, column=1, sticky="ns")

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("Keras model", "*.keras"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

    def browse_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select dataset folder")
        if path:
            self.dataset_path_var.set(path)

    def append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_busy(self, busy: bool, message: str | None = None) -> None:
        self.root.config(cursor="watch" if busy else "")
        if message:
            self.status_var.set(message)
        for child in self.root.winfo_children():
            self._set_widget_state_recursive(child, tk.DISABLED if busy else tk.NORMAL)
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()

    def _set_widget_state_recursive(self, widget: tk.Widget, state: str) -> None:
        try:
            if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Combobox)):
                widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, state)

    def run_async(self, target, busy_message: str) -> None:
        def worker() -> None:
            self.root.after(0, lambda: self.set_busy(True, busy_message))
            try:
                target()
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(0, lambda: self.append_log(tb))
                self.root.after(0, lambda: self.status_var.set(f"Error: {exc}"))
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
            finally:
                self.root.after(0, lambda: self.set_busy(False, self.status_var.get()))

        threading.Thread(target=worker, daemon=True).start()

    def load_model_async(self) -> None:
        self.run_async(self.load_model, "Loading model...")

    def load_model(self) -> None:
        model_path = Path(self.model_path_var.get().strip())
        dataset_path_raw = self.dataset_path_var.get().strip()
        dataset_path = dataset_path_raw if dataset_path_raw else None

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        input_shape = infer_input_shape()
        class_names = get_class_names(dataset_path, num_classes=DEFAULT_NUM_CLASSES)
        model = load_inference_model(
            model_path=model_path,
            input_shape=input_shape,
            num_classes=DEFAULT_NUM_CLASSES,
        )

        self.model = model
        self.class_names = class_names
        self.model_loaded = True

        self.root.after(0, lambda: self.append_log(f"Loaded model: {model_path}"))
        self.root.after(0, lambda: self.append_log(f"Recovered {len(class_names)} class names."))
        self.root.after(0, lambda: self.status_var.set("Model loaded successfully."))

    def ensure_model_loaded(self) -> None:
        if not self.model_loaded or self.model is None or self.class_names is None:
            self.load_model()

    def predict_wav_async(self) -> None:
        wav_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not wav_path:
            return
        self.run_async(lambda: self.predict_from_wav(Path(wav_path)), "Running WAV inference...")

    def predict_from_wav(self, wav_path: Path) -> None:
        self.ensure_model_loaded()
        if not wav_path.exists():
            raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

        pred_label, probs = predict_file(self.model, wav_path, self.class_names)
        self.root.after(0, lambda: self.show_prediction(wav_path, pred_label, probs))
        self.root.after(0, lambda: self.status_var.set("WAV inference complete."))

    def predict_mic_async(self) -> None:
        self.run_async(self.predict_from_mic, "Recording microphone input...")

    def predict_from_mic(self) -> None:
        self.ensure_model_loaded()

        try:
            duration = float(self.duration_var.get().strip())
        except ValueError as exc:
            raise ValueError("Mic duration must be a valid number, e.g. 2.0") from exc

        if duration <= 0:
            raise ValueError("Mic duration must be greater than 0.")

        temp_wav = Path.cwd() / "temp_gui_recording.wav"
        self.root.after(0, lambda: self.status_var.set(f"Recording for {duration:.1f}s... Speak now."))
        record_wav(temp_wav, duration=duration, sample_rate=TARGET_SAMPLE_RATE)

        pred_label, probs = predict_file(self.model, temp_wav, self.class_names)
        self.root.after(0, lambda: self.show_prediction(temp_wav, pred_label, probs))
        self.root.after(0, lambda: self.status_var.set("Microphone inference complete."))

    def show_prediction(self, wav_path: Path, pred_label: str, probs) -> None:
        self.input_var.set(f"Input: {wav_path}")
        self.prediction_var.set(f"Prediction: {pred_label}")
        self.append_log(f"Predicted '{pred_label}' for: {wav_path}")

        for item in self.tree.get_children():
            self.tree.delete(item)

        top_indices = list(reversed(probs.argsort()[-TOP_K:]))
        for rank, idx in enumerate(top_indices, start=1):
            label = self.class_names[idx]
            probability = float(probs[idx])
            self.tree.insert("", "end", values=(rank, label, f"{probability:.4f}"))

    def clear_results(self) -> None:
        self.input_var.set("Input: -")
        self.prediction_var.set("Prediction: -")
        self.status_var.set("Results cleared.")
        for item in self.tree.get_children():
            self.tree.delete(item)


def main() -> None:
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
