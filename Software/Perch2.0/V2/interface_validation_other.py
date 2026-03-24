import pandas as pd
import tkinter as tk
from tkinter import messagebox
import os
import pygame  # Pour jouer le son proprement

# --- CONFIGURATION ---
CSV_FILE = "rapport_anomalie_YAMNet_MAX.csv"
AUDIO_FOLDER = "Sound2test"


class AudioTinder:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Tinder - Validation d'Anomalies")
        self.root.geometry("500x400")

        # Initialisation de Pygame pour l'audio
        pygame.mixer.init()

        # Chargement des données
        if not os.path.exists(CSV_FILE):
            messagebox.showerror("Erreur", f"Le fichier {CSV_FILE} est introuvable.")
            root.destroy()
            return

        self.df = pd.read_csv(CSV_FILE)

        # On ajoute une colonne de validation si elle n'existe pas
        if 'Validation_Humaine' not in self.df.columns:
            self.df['Validation_Humaine'] = "A_VALIDER"

        # On ne garde que ceux qui ne sont pas encore validés
        self.pending_indices = self.df[self.df['Validation_Humaine'] == "A_VALIDER"].index.tolist()
        self.current_pos = 0

        # UI Elements
        self.label_info = tk.Label(root, text="Fichier :", font=("Helvetica", 12, "bold"))
        self.label_info.pack(pady=10)

        self.label_stats = tk.Label(root, text="", fg="blue")
        self.label_stats.pack()

        self.label_dist = tk.Label(root, text="", font=("Helvetica", 10, "italic"))
        self.label_dist.pack(pady=5)

        # Boutons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="✅ RAS", bg="green", fg="white", width=15, height=2,
                  command=lambda: self.vote("RAS")).grid(row=0, column=0, padx=10)

        tk.Button(btn_frame, text="⚠️ ANOMALIE", bg="red", fg="white", width=15, height=2,
                  command=lambda: self.vote("ANOMALIE")).grid(row=0, column=1, padx=10)

        tk.Button(root, text="🔄 REPLAY", bg="orange", width=20,
                  command=self.play_audio).pack(pady=10)

        self.update_view()

    def update_view(self):
        if self.current_pos < len(self.pending_indices):
            idx = self.pending_indices[self.current_pos]
            filename = self.df.loc[idx, 'Source Audio']
            dist = self.df.loc[idx, 'Distance_Max']
            status_ia = self.df.loc[idx, 'Status']

            self.label_info.config(text=f"Fichier : {filename}")
            self.label_dist.config(text=f"Distance IA : {dist:.4f} ({status_ia})")
            self.label_stats.config(text=f"Progression : {self.current_pos + 1} / {len(self.pending_indices)}")

            self.play_audio()
        else:
            messagebox.showinfo("Terminé", "Tous les fichiers ont été validés !")
            self.root.destroy()

    def play_audio(self):
        idx = self.pending_indices[self.current_pos]
        file_path = os.path.join(AUDIO_FOLDER, self.df.loc[idx, 'Source Audio'])
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        else:
            print(f"Fichier introuvable : {file_path}")

    def vote(self, choice):
        # Enregistre le choix
        idx = self.pending_indices[self.current_pos]
        self.df.at[idx, 'Validation_Humaine'] = choice

        # Sauvegarde immédiate dans le CSV
        self.df.to_csv(CSV_FILE, index=False)

        # Passe au suivant
        self.current_pos += 1
        self.update_view()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioTinder(root)
    root.mainloop()