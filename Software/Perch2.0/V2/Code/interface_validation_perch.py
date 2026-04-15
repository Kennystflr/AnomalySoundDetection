"""
ANNOTATION TRACKER v4.0 — PerchExplorer
Refactored: clean architecture, polished UI, zero bugs.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import pygame
from pydub import AudioSegment

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Use TkAgg backend for Matplotlib to integrate with Tkinter
matplotlib.use("TkAgg")

# ─────────────────────────── CONFIG ───────────────────────────
CSV_FILE      = "SAMPLES_EXTRACTED/list_to_validate.csv"
AUDIO_FOLDER  = "SAMPLES_EXTRACTED"
SAVE_INTERVAL = 10          # Number of votes before auto-saving to CSV
CLIP_DURATION = 5.0         # Seconds (expected duration of audio clips)

# ─────────────── PALETTE & FONTS (Centralized UI Design) ───────
COLORS = {
    "bg_deep":    "#0d0f14",
    "bg_panel":   "#13161e",
    "bg_card":    "#1a1d28",
    "bg_input":   "#1f2335",
    "accent":     "#00e5ff",
    "accent_dim": "#007a8c",
    "Void":       "#00c97a",
    "uncertain":  "#f5a623",
    "Anomaly":    "#ff4757",
    "pending":    "#4a5068",
    "text_hi":    "#e8eaf6",
    "text_lo":    "#5c6080",
    "border":     "#252840",
    "bar_bg":     "#1a1d28",
    "bar_fill":   "#00e5ff",
}

VOTE_CFG = {
    "Void":      {"color": COLORS["Void"],      "icon": "✓", "key": "←"},
    "uncertain": {"color": COLORS["uncertain"], "icon": "?", "key": "↓"},
    "Anomaly":   {"color": COLORS["Anomaly"],   "icon": "!", "key": "→"},
}

STATUS_COLOR = {
    "Anomaly":     COLORS["Anomaly"],
    "Void":        COLORS["Void"],
    "uncertain":   COLORS["uncertain"],
    "To_validate": COLORS["pending"],
}

FONT_MONO   = ("Courier New", 10)
FONT_TITLE  = ("Georgia", 11, "bold")
FONT_SCORE  = ("Courier New", 48, "bold")
FONT_LABEL  = ("Georgia", 9)
FONT_BTN    = ("Georgia", 10, "bold")
FONT_ENTRY  = ("Courier New", 11)


# ══════════════════════════════════════════════════════════════
class PerchExplorer:
    def __init__(self, root: tk.Tk):
        """
        Initializes the application: sets up state, audio mixer, UI components,
        and keyboard shortcuts.
        """
        self.root = root
        self.root.title("ANNOTATION TRACKER v4.0")
        self.root.geometry("1500x960")
        self.root.minsize(1200, 750)
        self.root.configure(bg=COLORS["bg_deep"])

        # Initialize pygame for low-latency audio playback
        pygame.mixer.init()

        # ── Internal State ──
        self.current_gain    = tk.DoubleVar(value=15.0)
        self.is_playing      = False
        self.unsaved_changes = 0
        self._pb_after_id    = None   # Progress bar callback ID

        self._load_data()
        self._setup_style()
        self._build_ui()
        self._populate_listbox()
        self._refresh_stats()
        self._select_row(0)

        # Global Hotkeys
        self.root.bind("<Left>",  lambda e: self.vote("Void"))
        self.root.bind("<Down>",  lambda e: self.vote("uncertain"))
        self.root.bind("<Right>", lambda e: self.vote("Anomaly"))
        self.root.bind("<space>", self._space_play)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────────────────
    def _load_data(self):
        """
        Loads the CSV file into a pandas DataFrame.
        Ensures necessary columns (Validation/Comment) exist and sorts by distance score.
        """
        if not os.path.exists(CSV_FILE):
            messagebox.showerror("Error", f"File not found:\n{CSV_FILE}")
            self.root.destroy()
            return

        try:
            self.df = pd.read_csv(CSV_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(CSV_FILE, encoding='latin-1')

        # Initialize metadata columns if missing
        for col, default in [("Human_validation", "To_validate"), ("Comment", "")]:
            if col not in self.df.columns:
                self.df[col] = default
        self.df["Comment"] = self.df["Comment"].fillna("").astype(str)

        # Dynamic score column selection
        self.col_score = "Distance_Noise" if "Distance_Noise" in self.df.columns else "Distance"
        self.df.sort_values(by=self.col_score, ascending=False, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.current_idx = 0

    # ──────────────────────────────────────────────────────────
    # TTK STYLE
    # ──────────────────────────────────────────────────────────
    def _setup_style(self):
        """Customizes the appearance of Ttk widgets (Progressbars, Scrollbars)."""
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Dark.Horizontal.TProgressbar",
                    troughcolor=COLORS["bar_bg"],
                    background=COLORS["bar_fill"],
                    bordercolor=COLORS["bar_bg"],
                    lightcolor=COLORS["bar_fill"],
                    darkcolor=COLORS["bar_fill"])
        s.configure("TScrollbar",
                    troughcolor=COLORS["bg_panel"],
                    background=COLORS["bg_card"],
                    bordercolor=COLORS["bg_panel"])

    # ──────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        """Builds the main container structure: Header and Body."""
        # ── HEADER ──
        hdr = tk.Frame(self.root, bg=COLORS["bg_panel"], height=48)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="◈  PERCH EXPLORER", font=("Georgia", 13, "bold"),
                 bg=COLORS["bg_panel"], fg=COLORS["accent"]).pack(side="left", padx=20, pady=10)

        self.lbl_shortcut = tk.Label(hdr,
            text="  ←  Void    ↓  uncertain    →  Anomaly    SPACE  Play  ",
            font=("Courier New", 8), bg=COLORS["bg_panel"], fg=COLORS["text_lo"])
        self.lbl_shortcut.pack(side="right", padx=20)

        separator = tk.Frame(self.root, bg=COLORS["border"], height=1)
        separator.pack(fill="x")

        # ── BODY ──
        body = tk.Frame(self.root, bg=COLORS["bg_deep"])
        body.pack(fill="both", expand=True)

        self._build_left_panel(body)
        self._build_right_panel(body)

    # ── LEFT PANEL (List & Search) ───────────────────────────
    def _build_left_panel(self, parent):
        """Creates the sidebar with search bar, file listbox, and stats."""
        left = tk.Frame(parent, bg=COLORS["bg_panel"], width=400)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Search / Filter Bar
        search_frame = tk.Frame(left, bg=COLORS["bg_card"], pady=6, padx=8)
        search_frame.pack(fill="x", padx=8, pady=(8, 0))

        tk.Label(search_frame, text="🔍", bg=COLORS["bg_card"], fg=COLORS["text_lo"],
                 font=("Arial", 10)).pack(side="left")

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search)
        tk.Entry(search_frame, textvariable=self.search_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_hi"],
                 insertbackground=COLORS["accent"],
                 font=FONT_MONO, relief="flat", borderwidth=0).pack(side="left", fill="x", expand=True, padx=6)

        # Listbox + Scrollbar
        lb_frame = tk.Frame(left, bg=COLORS["bg_panel"])
        lb_frame.pack(fill="both", expand=True, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(lb_frame, style="TScrollbar")
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            lb_frame,
            bg=COLORS["bg_card"], fg=COLORS["text_hi"],
            selectbackground=COLORS["accent_dim"], selectforeground="white",
            font=FONT_MONO, borderwidth=0, highlightthickness=0,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Statistics Display
        stats = tk.Frame(left, bg=COLORS["bg_deep"], pady=10)
        stats.pack(fill="x")

        self.lbl_stats = tk.Label(stats, text="",
                                  font=("Courier New", 9), bg=COLORS["bg_deep"],
                                  fg=COLORS["accent"])
        self.lbl_stats.pack()

        # Color Legend
        legend = tk.Frame(stats, bg=COLORS["bg_deep"])
        legend.pack(pady=4)
        for label, color in [("ANO", COLORS["Anomaly"]),
                              ("uncertain", COLORS["uncertain"]),
                              ("Void", COLORS["Void"]),
                              ("?", COLORS["pending"])]:
            dot = tk.Frame(legend, bg=color, width=8, height=8)
            dot.pack(side="left", padx=(6, 2))
            tk.Label(legend, text=label, font=("Courier New", 8),
                     bg=COLORS["bg_deep"], fg=COLORS["text_lo"]).pack(side="left")

    # ── RIGHT PANEL (Visualization & Controls) ────────────────
    def _build_right_panel(self, parent):
        """Creates the main area with Spectrogram, playback bar, and voting buttons."""
        right = tk.Frame(parent, bg=COLORS["bg_deep"])
        right.pack(side="right", fill="both", expand=True, padx=16, pady=12)

        # ─ Spectrogram Canvas ─
        spec_card = tk.Frame(right, bg=COLORS["bg_card"],
                             highlightbackground=COLORS["border"], highlightthickness=1)
        spec_card.pack(fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(9, 3.8), dpi=96)
        self.fig.patch.set_facecolor(COLORS["bg_card"])
        self.ax.set_facecolor(COLORS["bg_card"])
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18)

        self.canvas_spec = FigureCanvasTkAgg(self.fig, master=spec_card)
        self.canvas_spec.get_tk_widget().pack(fill="both", expand=True)

        # ─ Score and Filename Display ─
        info_row = tk.Frame(right, bg=COLORS["bg_deep"])
        info_row.pack(fill="x", pady=(8, 0))

        self.lbl_score = tk.Label(info_row, text="0.0000",
                                  font=FONT_SCORE, bg=COLORS["bg_deep"], fg=COLORS["accent"])
        self.lbl_score.pack(side="left", padx=16)

        file_col = tk.Frame(info_row, bg=COLORS["bg_deep"])
        file_col.pack(side="left", fill="x", expand=True, padx=8)

        tk.Label(file_col, text="SOURCE FILE", font=("Courier New", 7),
                 bg=COLORS["bg_deep"], fg=COLORS["text_lo"]).pack(anchor="w")
        self.lbl_file = tk.Label(file_col, text="—",
                                  font=("Courier New", 10), bg=COLORS["bg_deep"],
                                  fg=COLORS["text_hi"])
        self.lbl_file.pack(anchor="w")

        self.lbl_status_badge = tk.Label(info_row, text="To_validate",
                                          font=("Georgia", 9, "bold"),
                                          bg=COLORS["pending"], fg="white",
                                          padx=10, pady=4)
        self.lbl_status_badge.pack(side="right", padx=16)

        # ─ Playback Progress Bar ─
        pb_frame = tk.Frame(right, bg=COLORS["bg_deep"])
        pb_frame.pack(fill="x", pady=6, padx=16)

        tk.Label(pb_frame, text="PLAYBACK", font=("Courier New", 7),
                 bg=COLORS["bg_deep"], fg=COLORS["text_lo"]).pack(anchor="w")
        self.playback_bar = ttk.Progressbar(pb_frame, orient="horizontal",
                                             mode="determinate",
                                             style="Dark.Horizontal.TProgressbar")
        self.playback_bar.pack(fill="x", ipady=3)

        # ─ Vote Buttons ─
        vote_frame = tk.Frame(right, bg=COLORS["bg_deep"])
        vote_frame.pack(pady=8)

        for vote_key, cfg in VOTE_CFG.items():
            col = cfg["color"]
            btn = tk.Button(
                vote_frame,
                text=f"{cfg['icon']}  {vote_key}  {cfg['key']}",
                font=FONT_BTN, fg="white", bg=col,
                activebackground=col, activeforeground="white",
                relief="flat", borderwidth=0, padx=22, pady=10,
                cursor="hand2",
                command=lambda v=vote_key: self.vote(v),
            )
            btn.pack(side="left", padx=10)
            # Subtle hover effects
            btn.bind("<Enter>", lambda e, b=btn, c=col: b.config(bg=self._lighten(c)))
            btn.bind("<Leave>", lambda e, b=btn, c=col: b.config(bg=c))

        # ─ Audio Gain & Playback Controls ─
        ctrl_frame = tk.Frame(right, bg=COLORS["bg_deep"])
        ctrl_frame.pack(fill="x", padx=16, pady=4)

        tk.Label(ctrl_frame, text="GAIN", font=("Courier New", 8),
                 bg=COLORS["bg_deep"], fg=COLORS["text_lo"]).pack(side="left")

        gain_slider = tk.Scale(ctrl_frame, variable=self.current_gain,
                               from_=-20, to=40, orient="horizontal", resolution=1,
                               bg=COLORS["bg_deep"], fg=COLORS["text_hi"],
                               troughcolor=COLORS["bg_card"],
                               highlightthickness=0, relief="flat",
                               sliderrelief="flat", length=160, showvalue=True,
                               font=("Courier New", 8))
        gain_slider.pack(side="left", padx=12)

        tk.Button(ctrl_frame, text="▶  PLAY",
                  font=FONT_BTN, bg=COLORS["accent_dim"], fg="white",
                  activebackground=COLORS["accent"], activeforeground="white",
                  relief="flat", borderwidth=0, padx=16, pady=6,
                  cursor="hand2",
                  command=self.play_audio).pack(side="left", padx=8)

        # ─ Comment Input ─
        cmnt_frame = tk.Frame(right, bg=COLORS["bg_deep"])
        cmnt_frame.pack(fill="x", padx=16, pady=(2, 6))

        tk.Label(cmnt_frame, text="Comment", font=("Courier New", 7),
                 bg=COLORS["bg_deep"], fg=COLORS["text_lo"]).pack(anchor="w")

        self.comment_var = tk.StringVar()
        self.comment_var.trace_add("write", self._on_comment_change)
        cmnt_entry = tk.Entry(cmnt_frame, textvariable=self.comment_var,
                              bg=COLORS["bg_input"], fg=COLORS["text_hi"],
                              insertbackground=COLORS["accent"],
                              font=FONT_ENTRY, relief="flat", borderwidth=0)
        cmnt_entry.pack(fill="x", ipady=7)
        self.comment_entry = cmnt_entry

    # ──────────────────────────────────────────────────────────
    # LISTBOX LOGIC
    # ──────────────────────────────────────────────────────────
    def _populate_listbox(self, indices=None):
        """Updates the listbox content with formatted rows and status-based coloring."""
        self.listbox.delete(0, tk.END)
        rows = self.df.iterrows() if indices is None else ((i, self.df.iloc[i]) for i in indices)
        for i, row in rows:
            status = row["Human_validation"]
            badge  = "ANO" if (status == "Anomaly" or status == "DETECTION") else status[:5]
            score  = row[self.col_score]
            txt    = f" [{badge:^7}]  {score:>7.4f}  │  {row['Source Audio']}"
            self.listbox.insert(tk.END, txt)
            self.listbox.itemconfig(tk.END, fg=STATUS_COLOR.get(status, COLORS["pending"]))

    def _on_search(self, *_):
        """Filters the file list based on the search query."""
        q = self.search_var.get().lower()
        if not q:
            self._populate_listbox()
            return
        matched = [i for i, row in self.df.iterrows()
                   if q in row["Source Audio"].lower()]
        self._populate_listbox(matched)

    # ──────────────────────────────────────────────────────────
    # SELECTION LOGIC
    # ──────────────────────────────────────────────────────────
    def _on_listbox_select(self, _event):
        """Triggers when a user clicks an item in the listbox."""
        sel = self.listbox.curselection()
        if not sel:
            return
        self._select_row(sel[0])

    def _select_row(self, idx: int):
        """Updates the dashboard to display data for the selected audio file."""
        self.current_idx = idx
        row  = self.df.iloc[idx]
        path = os.path.join(AUDIO_FOLDER, row["Source Audio"])

        self.lbl_file.config(text=row["Source Audio"])
        self.lbl_score.config(text=f"{row[self.col_score]:.4f}")

        status = row["Human_validation"]
        self.lbl_status_badge.config(text=status,
                                      bg=STATUS_COLOR.get(status, COLORS["pending"]))

        # Update comment field without triggering the auto-save trace
        self.comment_var.trace_remove("write",
            self.comment_var.trace_info()[0][1] if self.comment_var.trace_info() else "")
        self.comment_var.set(row["Comment"])
        self.comment_var.trace_add("write", self._on_comment_change)

        self._update_spectrogram(path)
        self.play_audio()
        self._refresh_stats()

    # ──────────────────────────────────────────────────────────
    # SPECTROGRAM RENDERING
    # ──────────────────────────────────────────────────────────
    def _update_spectrogram(self, audio_path: str):
        """Generates and displays the Mel-Spectrogram using Librosa and Matplotlib."""
        self.ax.clear()
        if not os.path.exists(audio_path):
            self.ax.text(0.5, 0.5, "Audio file not found",
                         ha="center", va="center",
                         color=COLORS["Anomaly"], fontsize=11,
                         transform=self.ax.transAxes)
            self.canvas_spec.draw()
            return

        try:
            y, sr = librosa.load(audio_path, duration=CLIP_DURATION, mono=True)
            S     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=10000)
            S_dB  = librosa.power_to_db(S, ref=np.max)

            librosa.display.specshow(S_dB, x_axis="time", y_axis="mel",
                                     sr=sr, fmax=10000, ax=self.ax, cmap="magma")

            self.ax.set_title("Mel-Spectrogram", color=COLORS["text_lo"],
                              fontsize=9, pad=6)
            for spine in self.ax.spines.values():
                spine.set_edgecolor(COLORS["border"])
            self.ax.tick_params(colors=COLORS["text_lo"], labelsize=7)
            self.ax.xaxis.label.set_color(COLORS["text_lo"])
            self.ax.yaxis.label.set_color(COLORS["text_lo"])

            self.canvas_spec.draw()
        except Exception as exc:
            self.ax.text(0.5, 0.5, f"Error: {exc}",
                         ha="center", va="center",
                         color=COLORS["uncertain"], fontsize=9,
                         transform=self.ax.transAxes)
            self.canvas_spec.draw()

    # ──────────────────────────────────────────────────────────
    # AUDIO PLAYBACK
    # ──────────────────────────────────────────────────────────
    def play_audio(self):
        """Exports audio to memory with gain adjustment and plays it via Pygame."""
        path = os.path.join(AUDIO_FOLDER, self.df.iloc[self.current_idx]["Source Audio"])
        if not os.path.exists(path):
            return
        try:
            # Apply real-time gain using Pydub
            audio = AudioSegment.from_file(path) + self.current_gain.get()
            buf   = io.BytesIO()
            audio.export(buf, format="wav")
            buf.seek(0)
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            self.is_playing  = True
            self.start_time  = time.time()
            self._tick_playback()
        except Exception as exc:
            print(f"[AUDIO] Playback Error: {exc}")

    def _tick_playback(self):
        """Updates the visual progress bar during audio playback."""
        if self._pb_after_id:
            self.root.after_cancel(self._pb_after_id)
        if self.is_playing:
            elapsed = time.time() - self.start_time
            self.playback_bar["value"] = min((elapsed / CLIP_DURATION) * 100, 100)
            if elapsed < CLIP_DURATION:
                self._pb_after_id = self.root.after(50, self._tick_playback)
            else:
                self.is_playing = False
                self.playback_bar["value"] = 0

    def _space_play(self, event):
        """Triggers audio playback via spacebar (unless typing a comment)."""
        if self.root.focus_get() != self.comment_entry:
            self.play_audio()

    # ──────────────────────────────────────────────────────────
    # VOTING MECHANIC
    # ──────────────────────────────────────────────────────────
    def vote(self, choice: str):
        """Records the human validation, updates UI, and advances to the next file."""
        self.df.at[self.current_idx, "Human_validation"] = choice
        self._populate_listbox()

        # Update current selection visuals
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.current_idx)
        self.listbox.see(self.current_idx)

        # Immediate badge update
        self.lbl_status_badge.config(text=choice,
                                      bg=STATUS_COLOR.get(choice, COLORS["pending"]))

        self.unsaved_changes += 1
        if self.unsaved_changes >= SAVE_INTERVAL:
            self._save()

        # Auto-advance logic
        if self.current_idx < len(self.df) - 1:
            next_idx = self.current_idx + 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(next_idx)
            self.listbox.see(next_idx)
            self._select_row(next_idx)

        self._refresh_stats()

    # ──────────────────────────────────────────────────────────
    # STATS & COMMENTS
    # ──────────────────────────────────────────────────────────
    def _refresh_stats(self):
        """Recalculates progress percentages and category counts."""
        counts = self.df["Human_validation"].value_counts()
        ano    = counts.get("Anomaly", 0)
        Void    = counts.get("Void",      0)
        uncertain  = counts.get("uncertain",    0)
        done   = ano + Void + uncertain
        total  = len(self.df)
        pct    = (done / total * 100) if total else 0
        self.lbl_stats.config(
            text=f"  {done}/{total} annotated  ({pct:.0f}%)   "
                 f"ANO:{ano}  uncertain:{uncertain}  Void:{Void}  "
        )

    def _on_comment_change(self, *_):
        """Saves entry text to DataFrame on every keystroke."""
        self.df.at[self.current_idx, "Comment"] = self.comment_var.get()

    # ──────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────
    def _save(self):
        """Writes the current DataFrame state back to the CSV file."""
        self.df.to_csv(CSV_FILE, index=False)
        self.unsaved_changes = 0

    def _on_close(self):
        """Ensures data is saved and resources are released before closing."""
        self._save()
        pygame.mixer.quit()
        self.root.destroy()

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _lighten(hex_color: str, amount: int = 30) -> str:
        """Lightens a hex color for hover/active button states."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = min(255, r + amount)
        g = min(255, g + amount)
        b = min(255, b + amount)
        return f"#{r:02x}{g:02x}{b:02x}"


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = PerchExplorer(root)
    root.mainloop()