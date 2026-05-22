import numpy as np
import customtkinter as ctk
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Palet ───────────────────────────────────────────────────────────────────
_BG    = "#0d0d10"
_PANEL = "#111114"
_CARD  = "#18181c"
_BLUE  = "#3a86ff"
_RED   = "#ff453a"
_GREEN = "#30d158"
_TEXT  = "#f5f5f7"
_MUTED = "#636366"
_GRID  = "#2c2c2e"
_GOLD  = "#ffd60a"


def _style(ax, fig, title=""):
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=10, pad=8, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)


def _embed(fig, master):
    c = FigureCanvasTkAgg(fig, master=master)
    w = c.get_tk_widget()
    w.configure(bg=_BG, highlightthickness=0)
    w.pack(fill="both", expand=True)
    c.draw()
    return c


def _wrap(ctk_tab):
    """tk.Frame di dalam CTkTab agar canvas tidak hitam."""
    ctk_tab.configure(fg_color=_PANEL)
    f = tk.Frame(ctk_tab, bg=_BG)
    f.pack(fill="both", expand=True)
    return f


# ════════════════════════════════════════════════════════════════════════════
class VisualisasiWindow(ctk.CTkToplevel):

    def __init__(self, master, backend, threshold=5000):
        super().__init__(master)
        self.title("Visualisasi Nilai Eigen & Eigenface")
        self.geometry("980x660")
        self.resizable(True, True)
        self.configure(fg_color=_BG)
        self.backend   = backend
        self.threshold = threshold   # bisa di-override dari frontend

        if backend.eigenfaces is None or backend.mean_face is None:
            ctk.CTkLabel(self,
                text="⚠️  Dataset belum dimuat.\nLoad dataset terlebih dahulu.",
                font=("Arial", 13), text_color=_RED, justify="center"
            ).pack(expand=True)
            return

        self._build()

    # ── Header ───────────────────────────────────────────────────────────────
    def _build(self):
        hdr = ctk.CTkFrame(self, fg_color=_CARD, height=52, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr,
            text="📊  Visualisasi Nilai Eigen & Eigenface – ",
            font=("Arial", 13, "bold"), text_color=_TEXT
        ).pack(side="left", padx=18, pady=12)

        n   = len(self.backend.image_names)
        nef = self.backend.eigenfaces.shape[1]
        ctk.CTkLabel(hdr,
            text=f"Dataset: {n} gambar   Komponen: {nef}   "
                 f"Threshold: {self.threshold}   "
                 f"Ukuran: {self.backend.size[0]}×{self.backend.size[1]}",
            font=("Arial", 9), text_color=_MUTED
        ).pack(side="right", padx=18)

        # Threshold adjuster (agar bisa ubah tanpa buka ulang)
        ctrl = ctk.CTkFrame(self, fg_color=_CARD, height=36, corner_radius=0)
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)
        ctk.CTkLabel(ctrl, text="Threshold:", font=("Arial", 10),
                     text_color=_MUTED).pack(side="left", padx=(18, 4), pady=8)
        self._lbl_thr = ctk.CTkLabel(ctrl, text=str(self.threshold),
                                     font=("Arial", 10, "bold"), text_color=_RED)
        self._lbl_thr.pack(side="left", padx=(0, 8))
        sld = ctk.CTkSlider(ctrl, from_=1000, to=10000, number_of_steps=180,
                            width=200, height=14,
                            button_color=_RED, progress_color=_RED,
                            button_hover_color="#ff6961", fg_color=_GRID)
        sld.set(self.threshold)
        sld.pack(side="left", padx=4)
        sld.configure(command=self._on_thr)
        ctk.CTkLabel(ctrl, text="1000", font=("Arial", 8), text_color=_MUTED
                     ).pack(side="left", padx=2)
        ctk.CTkLabel(ctrl, text="10000", font=("Arial", 8), text_color=_MUTED
                     ).pack(side="left", padx=2)

        # TabView
        tv = ctk.CTkTabview(self, fg_color=_PANEL,
                            segmented_button_fg_color=_CARD,
                            segmented_button_selected_color=_BLUE,
                            segmented_button_selected_hover_color="#0066cc",
                            segmented_button_unselected_color=_CARD,
                            segmented_button_unselected_hover_color=_GRID,
                            text_color=_TEXT)
        tv.pack(fill="both", expand=True, padx=14, pady=(6, 12))

        for name in ["🖼  Eigenface",
                     "📊  Grafik Nilai Eigen",
                     "🧭  Visualisasi Vektor Eigen",
                     "📈  Plot Eigenface Pertama",
                     "🎯  Evaluasi Akurasi"]:
            tv.add(name)

        self._tab1  = _wrap(tv.tab("🖼  Eigenface"))
        self._tab2  = _wrap(tv.tab("📊  Grafik Nilai Eigen"))
        self._tab2b = _wrap(tv.tab("🧭  Visualisasi Vektor Eigen"))
        self._tab3  = _wrap(tv.tab("📈  Plot Eigenface Pertama"))
        self._tab4  = _wrap(tv.tab("🎯  Evaluasi Akurasi"))

        self._build_eigenface(self._tab1)
        self._build_norm_graph(self._tab2)
        self._build_eigen_vector_vis(self._tab2b)
        self._build_vector_plot(self._tab3)
        self._build_accuracy(self._tab4)

        # Simpan canvas tab 2 & 4 agar bisa di-refresh saat threshold berubah
        self._tv = tv

    # ── Threshold slider ─────────────────────────────────────────────────────
    def _on_thr(self, val):
        self.threshold = int(val)
        self._lbl_thr.configure(text=str(self.threshold))
        # Refresh tab Nilai Eigen dan Evaluasi
        for w in self._tab2.winfo_children():
            w.destroy()
        for w in self._tab4.winfo_children():
            w.destroy()
        self._build_norm_graph(self._tab2)
        self._build_accuracy(self._tab4)

    # ── Tab 1 : Mean Face + Eigenface grid ──────────────────────────────────
    def _build_eigenface(self, parent):
        h, w   = self.backend.size
        n_show = min(9, self.backend.eigenfaces.shape[1])
        total  = 1 + n_show
        cols   = min(5, total)
        rows   = (total + cols - 1) // cols

        fig = Figure(figsize=(cols * 1.9, rows * 2.1), dpi=90)
        fig.patch.set_facecolor(_BG)

        # Mean face
        ax0 = fig.add_subplot(rows, cols, 1)
        mf  = self.backend.mean_face.reshape(h, w)
        mf  = (mf - mf.min()) / (np.ptp(mf) + 1e-8)
        ax0.imshow(mf, cmap="gray", vmin=0, vmax=1)
        ax0.set_title("Mean Face", color=_TEXT, fontsize=8, pad=4)
        ax0.axis("off")

        for i in range(n_show):
            ax = fig.add_subplot(rows, cols, i + 2)
            ef = self.backend.eigenfaces[:, i].reshape(h, w)
            ef = (ef - ef.min()) / (np.ptp(ef) + 1e-8)
            ax.imshow(ef, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Eigenface #{i+1}", color=_MUTED, fontsize=7, pad=3)
            ax.axis("off")

        fig.tight_layout(pad=1)
        _embed(fig, parent)

    # ── Tab 2 : Grafik Nilai Eigen (eigenvalues) ────────────────────────────
    def _build_norm_graph(self, parent):
        ef  = self.backend.eigenfaces
        n   = ef.shape[1]
        idx = np.arange(1, n + 1)

        # Eigenvalue ≈ kuadrat norm vektor eigenface
        eigenvalues = np.linalg.norm(ef, axis=0) ** 2

        fig = Figure(figsize=(8.5, 4.4), dpi=90)
        ax  = fig.add_subplot(111)
        _style(ax, fig, title="Grafik Nilai Eigen (λ) Tiap Komponen PCA")

        ax.plot(idx, eigenvalues, color=_GOLD, lw=1.8,
                marker="o", ms=3, zorder=4, label="Nilai Eigen (λ)")

        thr_sq  = self.threshold ** 2
        ax.axhline(thr_sq, color=_RED, lw=1.4, ls="--",
                   label=f"Threshold² ({thr_sq:,})", zorder=3)

        n_above = int(np.sum(eigenvalues > thr_sq))
        if n_above > 0:
            ax.axvspan(0.5, n_above + 0.5, color=_BLUE, alpha=0.07,
                       label=f"Dominan: {n_above} komponen")

        ax.set_xlabel("Komponen ke-", fontsize=9)
        ax.set_ylabel("Nilai Eigen (λ)", fontsize=9)
        ax.set_xlim(0.5, n + 0.5)
        y_top = max(eigenvalues.max(), thr_sq) * 1.15
        ax.set_ylim(0, y_top)
        ax.grid(color=_GRID, lw=0.4, zorder=0)
        ax.legend(fontsize=7.5, facecolor=_CARD,
                  edgecolor=_GRID, labelcolor=_TEXT)

        ax.text(0.98, 0.97, "A·x = λ·x",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=_GOLD, fontweight="bold",
                bbox=dict(facecolor=_CARD, edgecolor=_GRID,
                          boxstyle="round,pad=0.35"))

        fig.tight_layout(pad=1.3)
        _embed(fig, parent)

    # ── Tab 2b : Visualisasi Vektor Eigen Ax=λx (4 kasus) ───────────────────
    def _build_eigen_vector_vis(self, parent):
        import matplotlib.patches as mpatches

        import matplotlib.patches as mpatches

        fig = Figure(figsize=(6.5, 5.6), dpi=90)
        fig.patch.set_facecolor(_BG)

        ax = fig.add_subplot(111)
        _style(ax, fig, title="Visualisasi Vektor Eigen: A·x = λ·x")

        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-2.3, 2.3)
        ax.set_aspect("equal")
        ax.axhline(0, color=_GRID, lw=0.6)
        ax.axvline(0, color=_GRID, lw=0.6)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.grid(color=_GRID, lw=0.3, alpha=0.4)

        x_vec = np.array([1.0, 1.0]) / np.sqrt(2)   # vektor asal (45°)

        cases = [
            (0.5,  _GREEN,    "(a) 0 ≤ λ ≤ 1  →  menyusut, arah sama"),
            (1.7,  _BLUE,     "(b) λ ≥ 1       →  memanjang, arah sama"),
            (-0.5, "#ff9f1c", "(c) -1 ≤ λ ≤ 0 →  menyusut, arah balik"),
            (-1.7, _RED,      "(d) λ ≤ -1      →  memanjang, arah balik"),
        ]

        # Vektor x asal (abu, digambar sekali)
        ax.annotate("", xy=x_vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=_MUTED,
                                    lw=1.4, mutation_scale=12))
        ax.text(x_vec[0] + 0.1, x_vec[1] + 0.08,
                "x", color=_MUTED, fontsize=10, fontweight="bold")

        legend_patches = [mpatches.Patch(color=_MUTED, label="x  (vektor asal)")]

        for lam, col, lbl in cases:
            lx_vec = lam * x_vec

            # Vektor λx
            ax.annotate("", xy=lx_vec, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=2.0, mutation_scale=13))

            # Label di ujung, offset menjauhi origin
            norm = np.linalg.norm(lx_vec) + 1e-8
            off  = lx_vec / norm * 0.28
            ax.text(lx_vec[0] + off[0], lx_vec[1] + off[1],
                    f"λ={lam}", color=col, fontsize=8.5,
                    ha="center", va="center",
                    bbox=dict(facecolor=_BG, edgecolor="none", pad=1.5))

            legend_patches.append(mpatches.Patch(color=col, label=lbl))

        # Titik origin
        ax.plot(0, 0, "o", color=_TEXT, ms=5, zorder=5)
        ax.text(0.06, -0.17, "O", color=_MUTED, fontsize=9)

        ax.legend(handles=legend_patches, fontsize=8, facecolor=_CARD,
                  edgecolor=_GRID, labelcolor=_TEXT,
                  loc="lower left", framealpha=0.92)

        fig.tight_layout(pad=1.4)
        _embed(fig, parent)

    # ── Tab 3 : Plot Vektor Eigenface Pertama (Gambar 4.5) ──────────────────
    def _build_vector_plot(self, parent):
        vec = self.backend.eigenfaces[:, 0]       # eigenface pertama
        px  = np.arange(len(vec))

        fig = Figure(figsize=(8.5, 4.2), dpi=90)
        ax  = fig.add_subplot(111)
        _style(ax, fig)

        ax.set_title("Plot Vektor Eigenface Pertama",
                     color=_TEXT, fontsize=10, fontweight="bold", pad=10)

        # Area chart (oranye/emas — sesuai tampilan di laporan)
        ax.fill_between(px, vec, 0, color="#ff9f1c", alpha=0.85, zorder=3)
        ax.plot(px, vec, color="#ffbf69", lw=0.4, zorder=4)
        ax.axhline(0, color=_GRID, lw=0.8, zorder=2)

        ax.set_xlabel("Index Piksel", fontsize=9)
        ax.set_ylabel("Nilai", fontsize=9)

        # Format sumbu X: 0, 2k, 4k, 6k, 8k, 10k
        total_px = len(vec)
        step     = max(total_px // 5, 1)
        xticks   = np.arange(0, total_px + 1, step)
        xlabels  = [f"{int(x/1000)}k" if x > 0 else "0" for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlim(0, total_px)
        ax.grid(axis="y", color=_GRID, lw=0.4, zorder=0)

        # Keterangan di bawah grafik (sesuai laporan Gambar 4.24)
        note = ("Penjelasan: Vektor eigen pertama menggambarkan pola variasi wajah "
                "paling dominan dalam dataset. Grafik ini memperlihatkan distribusi "
                "nilai dari komponen pertama yang dihasilkan PCA untuk setiap piksel "
                "pada gambar wajah.")
        fig.text(0.5, -0.04, note, ha="center", va="top",
                 color=_MUTED, fontsize=7, wrap=True,
                 bbox=dict(facecolor=_CARD, edgecolor=_GRID, boxstyle="round,pad=0.4"))

        fig.tight_layout(pad=1.2)
        _embed(fig, parent)

    # ── Tab 4 : Evaluasi Akurasi LOOCV + Confusion Matrix (Gambar 4.6) ─────
    def _build_accuracy(self, parent):
        # ── LOOCV di dalam visualisasi (tidak menyentuh backend) ──
        weights = self.backend.weights
        names   = self.backend.image_names
        n       = len(names)
        labels  = sorted(set(names))

        y_true, y_pred = [], []
        for i in range(n):
            mask  = np.ones(n, dtype=bool)
            mask[i] = False
            dists  = np.linalg.norm(weights[mask] - weights[i], axis=1)
            bidx   = np.argmin(dists)
            mdist  = dists[bidx]
            train_names = [names[j] for j in range(n) if j != i]
            y_true.append(names[i])
            y_pred.append(train_names[bidx] if mdist <= self.threshold else "Unknown")

        correct  = sum(t == p for t, p in zip(y_true, y_pred))
        accuracy = correct / n * 100

        # ── Confusion matrix ──
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        li = {l: i for i, l in enumerate(labels)}
        for t, p in zip(y_true, y_pred):
            if t in li and p in li:
                cm[li[t]][li[p]] += 1

        # ── Layout: info bar atas + grafik bawah ──
        info_bar = tk.Frame(parent, bg=_CARD, height=34)
        info_bar.pack(fill="x", padx=0, pady=(0, 4))
        info_bar.pack_propagate(False)

        tk.Label(info_bar, text="Melakukan evaluasi akurasi (LOOCV)…",
                 bg=_CARD, fg=_MUTED, font=("Arial", 9)
                 ).pack(side="left", padx=14, pady=6)

        acc_bar = tk.Frame(parent, bg="#14532d", height=32)
        acc_bar.pack(fill="x", padx=0, pady=(0, 6))
        acc_bar.pack_propagate(False)
        tk.Label(acc_bar, text=f"Akurasi Model: {accuracy:.2f}%",
                 bg="#14532d", fg=_GREEN, font=("Arial", 10, "bold")
                 ).pack(side="left", padx=14, pady=6)

        tk.Label(parent, text="Confusion Matrix",
                 bg=_BG, fg=_TEXT, font=("Arial", 10, "bold")
                 ).pack(anchor="w", padx=14, pady=(0, 2))

        graph_frame = tk.Frame(parent, bg=_BG)
        graph_frame.pack(fill="both", expand=True, padx=0, pady=0)

        fig_h = max(3.6, len(labels) * 0.9)
        fig   = Figure(figsize=(8.5, fig_h), dpi=90)
        ax    = fig.add_subplot(111)
        _style(ax, fig)

        # Heatmap manual dengan imshow
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, color=_TEXT)
        ax.set_yticklabels(labels, fontsize=9, color=_TEXT)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

        # Nilai di dalam cell
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                val   = cm[i, j]
                color = _BG if (val / vmax) > 0.5 else _TEXT
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color=_TEXT, labelsize=7)
        cbar.outline.set_edgecolor(_GRID)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color(_TEXT)

        fig.tight_layout(pad=1.2)
        _embed(fig, graph_frame)