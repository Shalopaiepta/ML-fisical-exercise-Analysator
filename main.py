# main.py
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import queue
from pose_analyzer import PoseAnalyzer
from datetime import datetime

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Тренер — Анализ приседаний")
        self.geometry("1280x800")
        self.analyzer = PoseAnalyzer()
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.session_error_counts = {}
        self.session_frames = 0
        self.session_start = 0
        self.user_name = ""

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

        ctk.CTkLabel(box, text="Вход в AI Тренер", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=30)
        ctk.CTkLabel(box, text="Введите данные для начала тренировки", font=ctk.CTkFont(size=14)).pack(pady=10)

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

        # Основная сетка
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)

        # Видео
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Правая панель
        side = ctk.CTkFrame(self)
        side.grid(row=0, column=1, sticky="nsew", padx=10, pady=20)

        ctk.CTkLabel(side, text=f"👤 {self.user_name}", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        self.status_label = ctk.CTkLabel(side, text="Статус: Ожидание", font=ctk.CTkFont(size=16))
        self.status_label.pack(pady=10)

        # === КРАСНАЯ РАМКА ОШИБОК ===
        err_title = ctk.CTkLabel(side, text="АНАЛИЗ ТЕХНИКИ", text_color="#FF4C4C", font=ctk.CTkFont(size=14, weight="bold"))
        err_title.pack(pady=(20,5))

        self.error_box = ctk.CTkFrame(side, border_color="#FF4C4C", border_width=3, corner_radius=10)
        self.error_box.pack(padx=15, pady=10, fill="both", expand=True)

        self.error_label = ctk.CTkLabel(self.error_box, text="Ошибок не обнаружено\nТехника в норме ✅", 
                                        text_color="#FF4C4C", justify="left", wraplength=280)
        self.error_label.pack(padx=15, pady=15)

        # Кнопки
        btn_frame = ctk.CTkFrame(side)
        btn_frame.pack(side="bottom", pady=30)

        self.start_btn = ctk.CTkButton(btn_frame, text="▶ Начать сессию", fg_color="green", command=self.start_recording)
        self.start_btn.pack(side="left", padx=10)

        self.stop_btn = ctk.CTkButton(btn_frame, text="■ Завершить", fg_color="red", command=self.stop_recording)
        self.stop_btn.pack(side="left", padx=10)

        ctk.CTkButton(side, text="Сохранить отчёт", command=self.save_report).pack(pady=10)

        # Запускаем камеру
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        self.update_ui()

    def video_loop(self):
        timestamp = 0
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                timestamp += 33
                rgb, label, conf, errors = self.analyzer.process_frame(frame, timestamp, self.is_recording)
                
                if self.is_recording:
                    for err in errors:
                        self.session_error_counts[err] = self.session_error_counts.get(err, 0) + 1
                    self.session_frames += 1

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

            self.status_label.configure(text=f"Действие: {label}  ({conf*100:.0f}%)")

            if errors:
                self.error_label.configure(text="\n".join([f"• {e}" for e in errors]))
            else:
                self.error_label.configure(text="Техника в норме ✅")
        except queue.Empty:
            pass

        self.after(10, self.update_ui)

    def start_recording(self):
        self.is_recording = True
        self.session_error_counts = {}
        self.session_frames = 0
        self.session_start = datetime.now()
        self.start_btn.configure(state="disabled")

    def stop_recording(self):
        self.is_recording = False
        self.start_btn.configure(state="normal")
        # Здесь можно показать итоговый отчёт в отдельном окне (реализовано в save_report)

    def save_report(self):
        # Ваш оригинальный generate_report
        duration = (datetime.now() - self.session_start).seconds if self.session_frames else 0
        report = self.analyzer.generate_report(self.session_error_counts, self.session_frames, duration)  # ваша функция
        os.makedirs("Reports", exist_ok=True)
        path = f"Reports/report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        ctk.CTkToplevel(self).title("Отчёт сохранён")
        ctk.CTkLabel(self, text=f"Отчёт сохранён в:\n{path}").pack(pady=20)

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()