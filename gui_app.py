"""SkinAura desktop interface powered by CustomTkinter."""

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import date, timedelta
import webbrowser

import customtkinter as ctk
from PIL import Image

from inference import get_model_warning, predict_skin_type_details
from recommend import (
    DISCLAIMER,
    PROFILE_OPTIONS,
    WEATHER_CONDITIONS,
    get_recommendation,
    get_weather_advice,
    resolve_profile,
)
import config


class SkinAuraApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SkinAura | Skin intelligence, made simple")
        self.geometry("1180x760")
        self.minsize(980, 680)
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.colors = {
            "ink": "#18212B", "muted": "#66737F", "paper": "#F7F8F6",
            "surface": "#FFFFFF", "line": "#E5E9E5", "green": "#2C765E",
            "green_light": "#E5F1EB", "peach": "#F8E9DE",
        }

        self.image_path = None
        self.preview_image = None
        self.is_analyzing = False
        self.current_skin_type = None
        self.current_recommendation = None
        self.predicted_skin_type = None
        self.profile_var = tk.StringVar(value=PROFILE_OPTIONS["image"])
        self.streak_count, self.streak_last_date = self._load_routine_streak()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.configure(fg_color=self.colors["paper"])
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=232, corner_radius=0, fg_color=self.colors["ink"])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        ctk.CTkLabel(sidebar, text="SkinAura", font=ctk.CTkFont("Georgia", 28, "bold"),
                     text_color="#FFFFFF").pack(anchor="w", padx=28, pady=(38, 2))
        ctk.CTkLabel(sidebar, text="AI SKIN INSIGHTS", font=ctk.CTkFont("Segoe UI", 10, "bold"),
                     text_color="#A7CABB").pack(anchor="w", padx=30, pady=(0, 52))
        self._nav_item(sidebar, "01  ANALYZE", True)
        self._nav_item(sidebar, "02  RECOMMENDATIONS", False)
        self._nav_item(sidebar, "03  ABOUT", False)
        ctk.CTkFrame(sidebar, height=1, fg_color="#35414A").pack(fill="x", padx=28, pady=(38, 20))
        ctk.CTkLabel(sidebar, text="PRIVATE BY DESIGN", font=ctk.CTkFont("Segoe UI", 10, "bold"),
                     text_color="#A7CABB").pack(anchor="w", padx=30)
        ctk.CTkLabel(sidebar, text="Your image is processed locally\nand is never uploaded.",
                     font=ctk.CTkFont("Segoe UI", 11), text_color="#CBD5D0", justify="left").pack(
                         anchor="w", padx=30, pady=(8, 0))

        content = ctk.CTkFrame(self, fg_color=self.colors["paper"], corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew", padx=34, pady=28)
        content.grid_columnconfigure(0, weight=5)
        content.grid_columnconfigure(1, weight=6)
        content.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(content, text="Understand your skin.", font=ctk.CTkFont("Georgia", 34, "bold"),
                     text_color=self.colors["ink"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ctk.CTkLabel(content, text="A considered starting point for a better daily routine.",
                     font=ctk.CTkFont("Segoe UI", 14), text_color=self.colors["muted"]).grid(
                         row=1, column=0, columnspan=2, sticky="w", pady=(4, 26))

        upload = ctk.CTkFrame(content, fg_color=self.colors["surface"], corner_radius=16,
                              border_width=1, border_color=self.colors["line"])
        upload.grid(row=2, column=0, sticky="nsew", padx=(0, 16))
        upload.grid_propagate(False)
        ctk.CTkLabel(upload, text="YOUR PHOTO", font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color=self.colors["green"]).pack(anchor="w", padx=24, pady=(22, 14))
        ctk.CTkLabel(upload, text="PROFILE FOR SUGGESTIONS", font=ctk.CTkFont("Segoe UI", 10, "bold"),
                     text_color=self.colors["muted"]).pack(anchor="w", padx=24, pady=(0, 4))
        ctk.CTkOptionMenu(upload, values=list(PROFILE_OPTIONS.values()), variable=self.profile_var).pack(
            fill="x", padx=24, pady=(0, 12)
        )
        self.image_label = ctk.CTkLabel(upload, text="Drop in a clear face photo\nfor a more useful reading",
                                        font=ctk.CTkFont("Segoe UI", 15), text_color=self.colors["muted"],
                                        fg_color=self.colors["peach"], corner_radius=12)
        self.image_label.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        button_row = ctk.CTkFrame(upload, fg_color="transparent")
        button_row.pack(fill="x", padx=24, pady=(0, 22))
        ctk.CTkButton(button_row, text="Choose image", command=self.browse_image, height=42,
                      corner_radius=9, fg_color=self.colors["green"], hover_color="#225B48",
                      font=ctk.CTkFont("Segoe UI", 13, "bold")).pack(side="left", fill="x", expand=True)
        self.analyze_btn = ctk.CTkButton(button_row, text="Analyze", command=self.analyze_image,
                                         height=42, corner_radius=9, state="disabled",
                                         fg_color=self.colors["ink"], hover_color="#2E3A45",
                                         font=ctk.CTkFont("Segoe UI", 13, "bold"))
        self.analyze_btn.pack(side="left", fill="x", expand=True, padx=(10, 0))

        results = ctk.CTkScrollableFrame(content, label_text="YOUR SKIN PROFILE",
                                         label_font=ctk.CTkFont("Segoe UI", 11, "bold"),
                                         fg_color=self.colors["surface"], corner_radius=16,
                                         border_width=1, border_color=self.colors["line"])
        results.grid(row=2, column=1, sticky="nsew")
        self.result_var = tk.StringVar(value="Ready when you are")
        self.confidence_var = tk.StringVar(value="Upload a photo to begin")
        ctk.CTkLabel(results, textvariable=self.result_var, font=ctk.CTkFont("Georgia", 28, "bold"),
                     text_color=self.colors["ink"], anchor="w").pack(fill="x", padx=24, pady=(24, 2))
        ctk.CTkLabel(results, textvariable=self.confidence_var, font=ctk.CTkFont("Segoe UI", 12),
                     text_color=self.colors["green"], anchor="w").pack(fill="x", padx=24)
        self.match_label = self._result_section(results, "PHOTO-MODEL MATCHES", "Upload different images to compare class scores.")
        self.model_warning_label = self._result_section(results, "MODEL STATUS", "Predictions are experimental until a valid model is trained.")
        self.summary_label = ctk.CTkLabel(results, text="Your personalized result will appear here after analysis.",
                                          font=ctk.CTkFont("Segoe UI", 13), text_color=self.colors["muted"],
                                          justify="left", anchor="w", wraplength=380)
        self.summary_label.pack(fill="x", padx=24, pady=(16, 20))
        self.product_label = self._result_section(results, "ROUTINE STARTERS", "Choose an image to see product categories.")
        product_links = ctk.CTkFrame(results, fg_color="transparent")
        product_links.pack(fill="x", padx=24, pady=(0, 12))
        self.product_a_button = ctk.CTkButton(product_links, text="Open cleanser", command=lambda: self._open_product("product_a_url"))
        self.product_a_button.pack(side="left", fill="x", expand=True)
        self.product_b_button = ctk.CTkButton(product_links, text="Open moisturizer", command=lambda: self._open_product("product_b_url"))
        self.product_b_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.tips_label = self._result_section(results, "DAILY NOTES", "Small, consistent steps make a difference.")
        self.hydration_label = self._result_section(results, "HYDRATION", "Sip water regularly according to thirst and your own needs.")
        self.diet_label = self._result_section(results, "BALANCED EATING", "A varied, balanced diet supports general wellbeing.")
        self.morning_tasks_label = self._result_section(results, "MORNING TASKS", "Your morning checklist will appear after analysis.")
        self.evening_tasks_label = self._result_section(results, "EVENING TASKS", "Your evening checklist will appear after analysis.")
        ctk.CTkLabel(results, text="TODAY'S WEATHER", font=ctk.CTkFont("Segoe UI", 10, "bold"),
                     text_color=self.colors["green"], anchor="w").pack(fill="x", padx=24, pady=(4, 2))
        self.weather_var = tk.StringVar(value=WEATHER_CONDITIONS["mild"])
        self.weather_menu = ctk.CTkOptionMenu(
            results,
            values=list(WEATHER_CONDITIONS.values()),
            variable=self.weather_var,
            command=self._update_weather_advice,
        )
        self.weather_menu.pack(fill="x", padx=24, pady=(2, 8))
        self.weather_label = self._result_section(results, "WEATHER NOTE", "Select today's weather for a routine note.")
        self.streak_label = self._result_section(results, "ROUTINE STREAK", "0 days")
        self.streak_button = ctk.CTkButton(results, text="Mark today's routine complete", command=self._complete_routine)
        self.streak_button.pack(fill="x", padx=24, pady=(0, 16))
        ctk.CTkLabel(content, text=DISCLAIMER, font=ctk.CTkFont("Segoe UI", 10),
                     text_color=self.colors["muted"], wraplength=900, justify="left").grid(
                         row=3, column=0, columnspan=2, sticky="w", pady=(18, 0))

    def _nav_item(self, parent, text, selected):
        ctk.CTkLabel(parent, text=text, height=38, anchor="w", padx=14,
                     font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color="#FFFFFF" if selected else "#82918A",
                     fg_color="#2C765E" if selected else "transparent",
                     corner_radius=8).pack(fill="x", padx=20, pady=3)

    def _result_section(self, parent, title, value):
        frame = ctk.CTkFrame(parent, fg_color="#F5F8F5", corner_radius=10)
        frame.pack(fill="x", padx=24, pady=(0, 12))
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont("Segoe UI", 10, "bold"),
                     text_color=self.colors["green"], anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        label = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont("Segoe UI", 12),
                             text_color=self.colors["ink"], anchor="w", justify="left", wraplength=380)
        label.pack(fill="x", padx=14, pady=(0, 14))
        return label

    def _load_routine_streak(self):
        try:
            with open(os.path.join(config.MODEL_DIR, "routine_streak.json"), encoding="utf-8") as streak_file:
                streak = json.load(streak_file)
            last_date = date.fromisoformat(streak.get("last_date", ""))
            count = int(streak.get("count", 0))
            if last_date not in (date.today(), date.today() - timedelta(days=1)):
                return 0, None
            return count, last_date
        except (OSError, ValueError, TypeError):
            return 0, None

    def _complete_routine(self):
        today = date.today()
        if self.streak_last_date == today:
            return
        self.streak_count = self.streak_count + 1 if self.streak_last_date == today - timedelta(days=1) else 1
        self.streak_last_date = today
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(os.path.join(config.MODEL_DIR, "routine_streak.json"), "w", encoding="utf-8") as streak_file:
            json.dump({"count": self.streak_count, "last_date": today.isoformat()}, streak_file)
        self._render_routine_streak()

    def _render_routine_streak(self):
        self.streak_label.configure(text=f"{self.streak_count} day(s) in a row")
        if self.streak_last_date == date.today():
            self.streak_button.configure(text="Today's routine is complete", state="disabled")

    def _open_product(self, product_url_key):
        if self.current_recommendation:
            webbrowser.open(self.current_recommendation[product_url_key])

    def _update_weather_advice(self, _condition):
        if self.current_skin_type:
            self.current_recommendation = get_recommendation(
                self.current_skin_type,
                self.weather_var.get(),
            )
            self.product_label.configure(
                text=(
                    f"1. {self.current_recommendation['product_a']}\n"
                    f"2. {self.current_recommendation['product_b']}"
                )
            )
            self.product_a_button.configure(text=self.current_recommendation["product_a"])
            self.product_b_button.configure(text=self.current_recommendation["product_b"])
            self.weather_label.configure(
                text=get_weather_advice(self.weather_var.get(), self.current_skin_type)
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return

        self.image_path = path
        img = Image.open(path).convert("RGB")
        self.preview_image = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))
        self.image_label.configure(image=self.preview_image, text="")
        self.analyze_btn.configure(state="normal")
        self.result_var.set("Ready to analyze")
        self.confidence_var.set("Your image is loaded")

    def analyze_image(self):
        if not self.image_path or self.is_analyzing:
            return
        self.is_analyzing = True
        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        self.result_var.set("Reading your image")
        self.confidence_var.set("The local model is working")
        self.profile_choice = next(
            key for key, label in PROFILE_OPTIONS.items() if label == self.profile_var.get()
        )
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            prediction_breakdown = predict_skin_type_details(self.image_path)
            predicted_label = prediction_breakdown[0]["label"]
            confidence = prediction_breakdown[0]["confidence"]
            profile, manual_profile = resolve_profile(self.profile_choice, predicted_label)
            rec = get_recommendation(profile)
            self.after(
                0,
                self._show_result,
                profile,
                confidence,
                rec,
                None,
                predicted_label,
                manual_profile,
                prediction_breakdown,
            )
        except Exception as error:
            self.after(0, self._show_result, None, None, None, error)

    def _show_result(
        self,
        label,
        confidence,
        rec,
        error,
        predicted_label=None,
        manual_profile=False,
        prediction_breakdown=None,
    ):
        self.is_analyzing = False
        self.analyze_btn.configure(state="normal", text="Analyze")
        if error:
            title = "Model unavailable" if isinstance(error, FileNotFoundError) else "Analysis failed"
            messagebox.showerror(title, str(error))
            self.result_var.set("Ready to try again")
            self.confidence_var.set("Check that the model has been trained")
            return
        self.result_var.set(label.title())
        self.confidence_var.set(
            f"Photo estimate: {predicted_label.title()} · {confidence:.1f}% confidence"
            if manual_profile
            else f"{confidence:.1f}% photo-model confidence"
        )
        self.model_warning_label.configure(text=get_model_warning() or "Updated transfer-learning model.")
        self.match_label.configure(
            text="\n".join(
                f"{match['label'].title()}: {match['confidence']:.1f}%"
                for match in prediction_breakdown
            )
        )
        self.current_skin_type = label
        self.predicted_skin_type = predicted_label or label
        self.current_recommendation = rec
        profile_note = f"Self-assessed profile. Photo model estimated {self.predicted_skin_type.title()}.\n\n" if manual_profile else ""
        self.summary_label.configure(text=profile_note + rec["summary"])
        self.product_label.configure(text=f"1. {rec['product_a']}\n2. {rec['product_b']}")
        self.tips_label.configure(text="\n".join(f"- {tip}" for tip in rec["tips"]))
        self.hydration_label.configure(text=rec["hydration_message"])
        self.diet_label.configure(text="\n".join(f"- {tip}" for tip in rec["diet_tips"]))
        self.morning_tasks_label.configure(text="\n".join(f"- {task}" for task in rec["morning_tasks"]))
        self.evening_tasks_label.configure(text="\n".join(f"- {task}" for task in rec["evening_tasks"]))
        self._update_weather_advice(self.weather_var.get())
        self._render_routine_streak()


if __name__ == "__main__":
    app = SkinAuraApp()
    app.mainloop()
