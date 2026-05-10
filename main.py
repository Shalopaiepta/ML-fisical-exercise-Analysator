import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import cv2
from PIL import Image

from pose_analyzer import PoseAnalyzer


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "Reports"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Тренер — анализ техники упражнений")
        self.geometry("1280x800")

        self.analyzer = PoseAnalyzer()
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.session_error_counts = {}
        self.session_label_counts = {}
        self.session_frames = 0
        self.session_start = None
        self.user_name = ""
        self.timestamp_counter = 0
        self.ui_loop_started = False

        self.video_queue = queue.Queue(maxsize=5)
        self.thread = None

        self.show_login()

    def show_login(self):
        for widget in self.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True)

        box = ctk.CTkFrame(frame, width=400, height=400, corner_radius=20)
        box.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            box,
            text="Вход в AI Тренер",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).pack(pady=30)
        ctk.CTkLabel(
            box,
            text="Введите данные для начала тренировки",
            font=ctk.CTkFont(size=14),
        ).pack(pady=10)

        self.username = ctk.CTkEntry(box, placeholder_text="Имя пользователя", width=300)
        self.username.pack(pady=12)

        self.password = ctk.CTkEntry(box, placeholder_text="Пароль", show="*", width=300)
        self.password.pack(pady=12)

        ctk.CTkButton(box, text="Войти", width=300, height=50, command=self.login).pack(pady=30)

    def login(self):
        self.user_name = self.username.get() or "Спортсмен"
        self.show_tracking()

    def show_tracking(self):
        for widget in self.winfo_children():
            widget.destroy()

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        side = ctk.CTkFrame(self)
        side.grid(row=0, column=1, sticky="nsew", padx=10, pady=20)

        ctk.CTkLabel(
            side,
            text=f"Пользователь: {self.user_name}",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=10)

        self.status_label = ctk.CTkLabel(side, text="Статус: ожидание", font=ctk.CTkFont(size=16))
        self.status_label.pack(pady=10)

        ref_title = ctk.CTkLabel(
            side,
            text="ЭТАЛОНЫ",
            text_color="#4CA6FF",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        ref_title.pack(pady=(15, 5))

        self.reference_status_label = ctk.CTkLabel(
            side,
            text="Проверяю состояние эталонов...",
            justify="left",
            wraplength=280,
        )
        self.reference_status_label.pack(padx=15, pady=(0, 10))

        ref_buttons = ctk.CTkFrame(side)
        ref_buttons.pack(padx=15, pady=(0, 10), fill="x")

        ctk.CTkButton(
            ref_buttons,
            text="Эталон приседаний",
            command=lambda: self.launch_reference_recorder("squats_reference.py"),
        ).pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkButton(
            ref_buttons,
            text="Эталон отжиманий",
            command=lambda: self.launch_reference_recorder("pushups_reference.py"),
        ).pack(fill="x", padx=10, pady=(0, 10))

        err_title = ctk.CTkLabel(
            side,
            text="АНАЛИЗ ТЕХНИКИ",
            text_color="#FF4C4C",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        err_title.pack(pady=(10, 5))

        self.error_box = ctk.CTkFrame(side, border_color="#FF4C4C", border_width=3, corner_radius=10)
        self.error_box.pack(padx=15, pady=10, fill="both", expand=True)

        self.error_label = ctk.CTkLabel(
            self.error_box,
            text="Ошибок не обнаружено\nТехника в норме",
            text_color="#FF4C4C",
            justify="left",
            wraplength=280,
        )
        self.error_label.pack(padx=15, pady=15)

        btn_frame = ctk.CTkFrame(side)
        btn_frame.pack(side="bottom", pady=30)

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="Начать сессию",
            fg_color="green",
            command=self.start_recording,
        )
        self.start_btn.pack(side="left", padx=10)

        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="Завершить",
            fg_color="red",
            command=self.stop_recording,
        )
        self.stop_btn.pack(side="left", padx=10)

        ctk.CTkButton(side, text="Сохранить отчёт", command=self.save_report).pack(pady=10)

        self.analyzer.reset_runtime_state()
        self.refresh_reference_status()
        self.start_video_stream()
        if not self.ui_loop_started:
            self.ui_loop_started = True
            self.update_ui()

    def start_video_stream(self):
        if self.is_running:
            return

        self.timestamp_counter = 0
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.status_label.configure(text="Статус: камера недоступна")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def stop_video_stream(self):
        self.is_running = False

        cap = self.cap
        self.cap = None
        if cap:
            cap.release()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

        while not self.video_queue.empty():
            try:
                self.video_queue.get_nowait()
            except queue.Empty:
                break

    def refresh_reference_status(self):
        squat_ready = self.analyzer.reference_ready("squats")
        pushup_ready = self.analyzer.reference_ready("pushups")

        squat_status = "готов" if squat_ready else "не записан"
        pushup_status = "готов" if pushup_ready else "не записан"
        note = ""
        if not pushup_ready:
            note = "\nОтжимания пока анализируются по защитным порогам. Запишите эталон для точной персональной проверки."

        self.reference_status_label.configure(
            text=(
                f"Приседания: {squat_status}\n"
                f"Отжимания: {pushup_status}"
                f"{note}"
            )
        )

    def show_message(self, title, text):
        window = ctk.CTkToplevel(self)
        window.title(title)
        ctk.CTkLabel(window, text=text, justify="left", wraplength=420).pack(padx=20, pady=20)

    def launch_reference_recorder(self, script_name):
        script_path = PROJECT_DIR / script_name
        if not script_path.exists():
            self.show_message("Скрипт не найден", f"Не удалось найти {script_path.name}.")
            return

        if self.is_recording:
            self.stop_recording()

        self.stop_video_stream()
        self.status_label.configure(text="Статус: запись эталона запущена")
        self.update_idletasks()

        result = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_DIR), check=False)

        self.analyzer.reload_reference_stats()
        self.refresh_reference_status()
        self.start_video_stream()

        family = "pushups" if "pushups" in script_name else "squats"
        family_name = "отжиманий" if family == "pushups" else "приседаний"

        if result.returncode == 0 and self.analyzer.reference_ready(family):
            self.show_message(
                "Эталон обновлён",
                f"Эталон для {family_name} готов. Следующие сессии будут использовать CSV-референс.",
            )
        elif result.returncode != 0:
            self.show_message(
                "Запись эталона завершилась с ошибкой",
                f"{script_path.name} завершился с кодом {result.returncode}.",
            )

    def video_loop(self):
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.timestamp_counter += 33
                rgb, label, conf, errors = self.analyzer.process_frame(
                    frame,
                    self.timestamp_counter,
                    self.is_recording,
                )

                if self.is_recording and label in self.analyzer.SUPPORTED_PHASES:
                    self.session_label_counts[label] = self.session_label_counts.get(label, 0) + 1
                    self.session_frames += 1
                    for err in errors:
                        self.session_error_counts[err] = self.session_error_counts.get(err, 0) + 1

                try:
                    self.video_queue.put_nowait((rgb, label, conf, errors))
                except queue.Full:
                    pass

            cv2.waitKey(1)

    def update_ui(self):
        try:
            rgb, label, conf, errors = self.video_queue.get_nowait()
            img = Image.fromarray(rgb)
            photo = ctk.CTkImage(light_image=img, size=(800, 600))
            self.video_label.configure(image=photo)
            self.video_label.image = photo

            pretty_label = self.analyzer.pretty_label(label)
            self.status_label.configure(text=f"Действие: {pretty_label} ({conf * 100:.0f}%)")

            if errors:
                self.error_label.configure(text="\n".join([f"• {error}" for error in errors]))
            elif self.is_recording and label in self.analyzer.SUPPORTED_PHASES:
                self.error_label.configure(text="Техника в норме")
            else:
                self.error_label.configure(text="Начните сессию для анализа ошибок")
        except queue.Empty:
            pass

        self.after(10, self.update_ui)

    def start_recording(self):
        self.is_recording = True
        self.session_error_counts = {}
        self.session_label_counts = {}
        self.session_frames = 0
        self.session_start = datetime.now()
        self.analyzer.reset_session_state()
        self.start_btn.configure(state="disabled")

    def stop_recording(self):
        self.is_recording = False
        self.analyzer.reset_session_state()
        self.start_btn.configure(state="normal")

    def save_report(self):
        if not self.session_frames or not self.session_start:
            self.show_message("Нет данных", "Сначала проведите сессию записи, затем сохраните отчёт.")
            return

        duration = (datetime.now() - self.session_start).seconds
        report = self.analyzer.generate_report(
            self.session_error_counts,
            self.session_frames,
            duration,
            self.session_label_counts,
        )

        family = self.analyzer.detect_session_family(self.session_label_counts) or "exercise"
        prefix = {
            "squats": "squat",
            "pushups": "pushup",
            "mixed": "exercise",
            "exercise": "exercise",
        }.get(family, "exercise")

        REPORTS_DIR.mkdir(exist_ok=True)
        path = REPORTS_DIR / f"{prefix}_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with path.open("w", encoding="utf-8") as file:
            file.write(report)

        self.show_message("Отчёт сохранён", f"Отчёт сохранён в:\n{path}")

    def on_closing(self):
        self.stop_video_stream()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
