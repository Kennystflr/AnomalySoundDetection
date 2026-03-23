import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import os
import pygame
from pydub import AudioSegment
import io
import time

# --- CONFIGURATION ---
CSV_FILE = "rapport_anomalies_optimize.csv"
AUDIO_FOLDER = "Sound2test"


class PerchExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 ANNOTATION TRACKER v3 - STATS & COMMENTS")
        self.root.geometry("1400x950")
        self.root.configure(bg="#121212")

        pygame.mixer.init()

        self.current_gain = tk.DoubleVar(value=15.0)
        self.is_playing = False
        self.start_time = 0

        if not os.path.exists(CSV_FILE):
            messagebox.showerror("Erreur", f"Fichier {CSV_FILE} introuvable.")
            root.destroy()
            return

        self.df = pd.read_csv(CSV_FILE)

        # Initialisation des colonnes
        if 'Validation_Humaine' not in self.df.columns:
            self.df['Validation_Humaine'] = "A_VALIDER"
        if 'Commentaire' not in self.df.columns:
            self.df['Commentaire'] = ""

        if 'Distance_Noise' in self.df.columns:
            self.col_primary = 'Distance_Noise'
        else:
            self.col_primary = 'Distance'

        self.df = self.df.sort_values(by=self.col_primary, ascending=False).reset_index(drop=True)
        self.current_idx = 0

        self.setup_ui()
        self.update_listbox()
        self.update_stats()

        # Raccourcis
        self.root.bind("<Left>", lambda e: self.vote("RAS"))
        self.root.bind("<Down>", lambda e: self.vote("DOUTE"))
        self.root.bind("<Right>", lambda e: self.vote("ANOMALIE"))
        self.root.bind("<space>", lambda e: self.play_audio())

    def setup_ui(self):
        # --- PANNEAU GAUCHE ---
        self.left_frame = tk.Frame(self.root, bg="#1e1e1e", width=550)
        self.left_frame.pack(side="left", fill="y", padx=5, pady=5)

        tk.Label(self.left_frame, text="LISTE DES SAMPLES", bg="#1e1e1e", fg="gray", font=("Arial", 9, "bold")).pack(
            pady=10)

        self.list_container = tk.Frame(self.left_frame, bg="#1e1e1e")
        self.list_container.pack(fill="both", expand=True, padx=5)

        self.scrollbar = tk.Scrollbar(self.list_container)
        self.scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(self.list_container, bg="#181818", fg="#e0e0e0", font=("Consolas", 10),
                                  width=75, selectbackground="#333333", borderwidth=0,
                                  yscrollcommand=self.scrollbar.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_file)

        # --- NOUVEAU : PANNEAU DE STATISTIQUES (Bas Gauche) ---
        self.stats_frame = tk.LabelFrame(self.left_frame, text=" 📊 DASHBOARD ", bg="#252525", fg="white",
                                         font=("Arial", 10, "bold"), padx=15, pady=15)
        self.stats_frame.pack(side="bottom", fill="x", padx=10, pady=20)

        # Labels pour les stats
        self.label_total = tk.Label(self.stats_frame, text="Total: 0/0", bg="#252525", fg="#00ffcc",
                                    font=("Arial", 11, "bold"))
        self.label_total.grid(row=0, column=0, sticky="w", pady=5)

        self.label_ano = tk.Label(self.stats_frame, text="Anomalies: 0", bg="#252525", fg="#e74c3c", font=("Arial", 10))
        self.label_ano.grid(row=1, column=0, sticky="w")

        self.label_ras = tk.Label(self.stats_frame, text="RAS: 0", bg="#252525", fg="#2ecc71", font=("Arial", 10))
        self.label_ras.grid(row=1, column=1, sticky="w", padx=20)

        self.label_dou = tk.Label(self.stats_frame, text="Doutes: 0", bg="#252525", fg="#f1c40f", font=("Arial", 10))
        self.label_dou.grid(row=1, column=2, sticky="w")

        # --- PANNEAU DROIT ---
        self.right_frame = tk.Frame(self.root, bg="#121212")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=30)

        self.label_file = tk.Label(self.right_frame, text="Sélectionnez un fichier", font=("Arial", 12), bg="#121212",
                                   fg="white")
        self.label_file.pack(pady=20)

        self.val_d1 = tk.Label(self.right_frame, text="0.0000", font=("Impact", 70), bg="#121212", fg="#00ffcc")
        self.val_d1.pack(pady=10)

        self.label_comm = tk.Label(self.right_frame, text="", font=("Arial", 12, "italic"), bg="#121212", fg="#f1c40f",
                                   wraplength=500)
        self.label_comm.pack(pady=20)

        self.progress_frame = tk.Frame(self.right_frame, bg="#121212")
        self.progress_frame.pack(fill="x", pady=20)
        self.playback_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=400, mode="determinate")
        self.playback_bar.pack(pady=5)

        self.gain_frame = tk.Frame(self.right_frame, bg="#121212")
        self.gain_frame.pack(pady=20)
        tk.Label(self.gain_frame, text="GAIN (dB) :", bg="#121212", fg="gray").pack(side="left")
        ttk.Scale(self.gain_frame, from_=0, to=40, variable=self.current_gain, orient="horizontal", length=200).pack(
            side="left", padx=10)

        self.btn_frame = tk.Frame(self.right_frame, bg="#121212")
        self.btn_frame.pack(pady=40)
        for text, color, choice in [("✅ RAS", "#2ecc71", "RAS"), ("❓ DOUTE", "#f1c40f", "DOUTE"),
                                    ("⚠️ ANO", "#e74c3c", "ANOMALIE")]:
            tk.Button(self.btn_frame, text=text, bg=color, fg="white", font=("Arial", 12, "bold"), width=12, height=3,
                      relief="flat", command=lambda c=choice: self.vote(c)).pack(side="left", padx=15)

        tk.Button(self.right_frame, text="🔊 RÉÉCOUTER (Espace)", font=("Arial", 12), bg="#3498db", fg="white", width=30,
                  height=2, command=self.play_audio).pack()

    def update_stats(self):
        total = len(self.df)
        counts = self.df['Validation_Humaine'].value_counts()

        ano = counts.get('ANOMALIE', 0)
        ras = counts.get('RAS', 0)
        dou = counts.get('DOUTE', 0)
        done = ano + ras + dou
        percent = int((done / total) * 100) if total > 0 else 0

        self.label_total.config(text=f"PROGRESSION GLOBAL : {done} / {total} ({percent}%)")
        self.label_ano.config(text=f"🔥 Anomalies : {ano}")
        self.label_ras.config(text=f"🌿 RAS : {ras}")
        self.label_dou.config(text=f"🤔 Doutes : {dou}")

    def update_listbox(self):
        pos = self.listbox.yview()
        self.listbox.delete(0, tk.END)
        for i, row in self.df.iterrows():
            label = row['Validation_Humaine']
            short_label = "ANO" if label == "ANOMALIE" else label
            prefix = f"[{short_label:^7}]"
            txt = f" {prefix} D:{row[self.col_primary]:.3f} | {row['Source Audio']}"
            self.listbox.insert(tk.END, txt)

            if label == "ANOMALIE":
                color = "#e74c3c"
            elif label == "RAS":
                color = "#2ecc71"
            elif label == "DOUTE":
                color = "#f1c40f"
            else:
                color = "#555555"
            self.listbox.itemconfig(i, fg=color)
        self.listbox.yview_moveto(pos[0])

    def vote(self, choice):
        comment = self.df.at[self.current_idx, 'Commentaire']  # Garde l'ancien par défaut
        if choice == "DOUTE":
            comment = simpledialog.askstring("Note de Doute", "Pourquoi ce doute ?", parent=self.root)
            if comment is None: return  # Annulation si on clique sur 'cancel'

        self.df.at[self.current_idx, 'Validation_Humaine'] = choice
        self.df.at[self.current_idx, 'Commentaire'] = comment if choice == "DOUTE" else ""
        self.df.to_csv(CSV_FILE, index=False)

        self.update_listbox()
        self.update_stats()

        if self.current_idx < len(self.df) - 1:
            self.current_idx += 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_idx)
            self.listbox.see(self.current_idx)
            self.on_select_file(None)

    def on_select_file(self, event):
        if self.listbox.curselection():
            self.current_idx = self.listbox.curselection()[0]
            row = self.df.iloc[self.current_idx]
            self.label_file.config(text=row['Source Audio'])
            self.val_d1.config(text=f"{row[self.col_primary]:.4f}")

            comm_text = f"📝 Note : {row['Commentaire']}" if pd.notna(row['Commentaire']) and row[
                'Commentaire'] != "" else ""
            self.label_comm.config(text=comm_text)
            self.play_audio()

    def play_audio(self):
        try:
            row = self.df.iloc[self.current_idx]
            path = os.path.join(AUDIO_FOLDER, row['Source Audio'])
            if os.path.exists(path):
                audio = AudioSegment.from_file(path) + self.current_gain.get()
                buf = io.BytesIO()
                audio.export(buf, format="wav")
                buf.seek(0)
                pygame.mixer.music.load(buf)
                pygame.mixer.music.play()
                self.is_playing, self.start_time = True, time.time()
                self.update_playback_bar()
        except Exception:
            pass

    def update_playback_bar(self):
        if self.is_playing:
            elapsed = time.time() - self.start_time
            if elapsed <= 5.0:
                self.playback_bar['value'] = (elapsed / 5.0) * 100
                self.root.after(50, self.update_playback_bar)
            else:
                self.playback_bar['value'] = 100
                self.is_playing = False


if __name__ == "__main__":
    root = tk.Tk()
    app = PerchExplorer(root)
    root.mainloop()