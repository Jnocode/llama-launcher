#!/usr/bin/env python3
"""
llama.cpp Expert Launcher (Tkinter GUI)
功能：硬體偵測、模型庫掃描、資源預估、收藏參數、API 整合狀態
環境：Windows + NVIDIA GPU
作者：AI Copilot
"""

import os
import sys
import subprocess
import json
import struct
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ==================== 環境路徑 ====================
LLAMA_DIR = r"D:\Workspace\artifacts\llama.cpp\b9060-cuda13.1"
LLAMA_SERVER = os.path.join(LLAMA_DIR, "llama-server.exe")
MODEL_DIR = r"D:\Workspace\artifacts\models"
LOG_DIR = r"D:\Workspace\artifacts\logs"
WORK_DIR = r"D:\Workspace"
DEFAULT_PORT = 8080
PRESETS_FILE = os.path.join(os.path.dirname(__file__), "launcher_presets.json")

# ==================== 硬體偵測 ====================
def detect_hardware():
    """自動偵測系統硬體資訊"""
    info = {"cpu": "Unknown", "ram_gb": 0, "gpu": "Unknown", "cuda": "Unknown", "vram_gb": 0}
    
    # CPU
    try:
        info["cpu"] = os.popen("wmic cpu get Name 2>nul | findstr /v Name").read().strip() or "Unknown"
    except Exception:
        pass
    
    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            out = os.popen("wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value 2>nul").read()
            for line in out.splitlines():
                if line.startswith("TotalVisibleMemorySize"):
                    kb = int(line.split("=")[1].strip())
                    info["ram_gb"] = round(kb / (1024 * 1024), 1)
        except Exception:
            pass
    
    # GPU & VRAM
    try:
        out = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits").read().strip()
        if out:
            parts = out.split(",")
            info["gpu"] = parts[0].strip()
            info["vram_gb"] = int(parts[1].strip()) // 1024
    except Exception:
        pass

    # CPU 實體核心數偵測 (為執行緒預設值做準備)
    try:
        cores_out = os.popen("wmic cpu get NumberOfCores /value 2>nul").read().strip()
        cores_sum = 0
        for line in cores_out.splitlines():
            if "NumberOfCores" in line:
                cores_sum += int(line.split("=")[1].strip())
        info["physical_cores"] = cores_sum if cores_sum > 0 else (os.cpu_count() // 2 if os.cpu_count() else 8)
    except Exception:
        info["physical_cores"] = os.cpu_count() // 2 if os.cpu_count() else 8
    
    # CUDA version from llama.cpp
    try:
        out = os.popen(f"{LLAMA_DIR}\\llama-server.exe --version 2>&1").read()
        if "cuda" in out.lower():
            info["cuda"] = "13.1"
        else:
            info["cuda"] = "N/A"
    except Exception:
        info["cuda"] = "N/A"
    
    return info


def estimate_resources(ctx_tokens, moe_cpu, ngl, kt, vt, vram_gb, ram_gb, mmproj_gb=0):
    """
    預估 VRAM / RAM 使用量
    基於 Qwen3.6-35B-A3B (MoE, Q4_K_M) 的實測數據校準
    
    實測基準點 (RTX 4070 12GB, llama-server):
    - n-cpu-moe 32 + 256K ctx + q4_0 → ~7.8 GB VRAM (64%)
    - n-cpu-moe 24 + 32K ctx + q4_0 → ~9.7 GB VRAM (79%)
    
    校準公式: VRAM = base + moe_on_gpu * coeff + ctx_blocks * kv_coeff + mmproj
    解方程:
      7.8 = base + 9*coeff + 8*kv   [moe=32, ctx=256K]
      9.7 = base + 17*coeff + 1*kv   [moe=24, ctx=32K]
    已知 kv(q4_0) ≈ 0.28 GB/32K blocks
    → coeff = 0.48 GB/layer, base = 1.22 GB
    """
    moe_cpu = int(moe_cpu)
    ctx_tokens = int(ctx_tokens)
    
    # === 校準常數 ===
    base_vram = 1.22        # embedding + 非 MoE 層 ≈ 1.2 GB
    moe_coeff = 0.48        # 每層 expert 上 GPU 增加 ≈ 480 MB VRAM
    kv_coeff_q4 = 0.28      # q4_0 KV cache ≈ 280 MB per 32K blocks
    
    moe_layers_total = 41
    moe_on_gpu = max(0, moe_layers_total - moe_cpu)
    moe_vram = moe_on_gpu * moe_coeff
    
    # === KV Cache VRAM ===
    ctx_blocks = ctx_tokens / 32768
    if kt == "q4_0" or vt == "q4_0":
        kv_vram = ctx_blocks * kv_coeff_q4
        kv_ram_backup = ctx_blocks * 0.15  # RAM backup for q4_0
    elif kt == "q8_0" or vt == "q8_0":
        kv_vram = ctx_blocks * kv_coeff_q4 * 2
        kv_ram_backup = ctx_blocks * 0.3
    else:  # f16
        kv_vram = ctx_blocks * 1.1
        kv_ram_backup = ctx_blocks * 0.8
    
    # === mmproj VRAM (多模態投影層) ===
    # BF16 mmproj 檔案 4GB 但實測僅 ~2.1 GB 上 VRAM
    mmproj_vram = mmproj_gb if mmproj_gb > 0 else 0
    
    # === 總計 ===
    total_vram = base_vram + moe_vram + kv_vram + mmproj_vram
    moe_ram_backup = (moe_layers_total - moe_on_gpu) * 0.5  # GB RAM per CPU expert layer
    total_ram = moe_ram_backup + kv_ram_backup + 2.0  # +2GB overhead
    
    # === 狀態判斷 ===
    vram_free = vram_gb - total_vram
    ram_free = ram_gb - total_ram
    
    if total_vram > vram_gb:
        status = "🔴 爆顯存"
        status_color = "red"
    elif vram_free < 1.0:
        status = "🟡 危險"
        status_color = "orange"
    elif total_vram > vram_gb * 0.85:
        status = "🟠 高負載"
        status_color = "orange"
    else:
        status = "🟢 安全"
        status_color = "green"
    
    return {
        "vram_gb": round(total_vram, 1),
        "ram_gb": round(total_ram, 1),
        "kv_vram_gb": round(kv_vram, 1),
        "vram_free_gb": round(max(0, vram_free), 1),
        "ram_free_gb": round(max(0, ram_free), 1),
        "status": status,
        "status_color": status_color,
    }


# ==================== 收藏參數載入 ====================
def load_favorites():
    """載入收藏參數"""
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_favorite(name, params):
    """儲存收藏參數"""
    favorites = load_favorites()
    # 檢查是否已存在
    for i, fav in enumerate(favorites):
        if fav["name"] == name:
            favorites[i] = {"name": name, **params}
            break
    else:
        favorites.append({"name": name, **params})
    
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(favorites, f, ensure_ascii=False, indent=2)
    return favorites


# ==================== 模型庫掃描 ====================
def scan_models_detailed():
    """掃描模型並回傳詳細資訊（含 mmproj 對應）"""
    models = []
    mmproj_map = {}  # {model_base_name: mmproj_path}
    
    if not os.path.isdir(MODEL_DIR):
        return models, mmproj_map
    
    # 第一次掃描：收集所有 mmproj
    mmproj_files = {}
    for root_dir, _, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith(".gguf") and "mmproj" in f.lower():
                full = os.path.join(root_dir, f)
                size_gb = os.path.getsize(full) / (1024**3)
                # 從 mmproj 檔名提取對應的模型名稱
                # 例如: Qwen3.6-35B-A3B-mmproj-BF16.gguf → Qwen3.6-35B-A3B
                base = f.replace("-mmproj-", "-").replace("_mmproj_", "_").replace(".gguf", "")
                # 去掉量化後綴
                for q in ["BF16", "F16", "F32", "Q8_0", "Q4_K_M", "Q5_K_M", "Q6_K"]:
                    if q in base:
                        base = base.replace(f"-{q}", "").replace(f"_{q}", "")
                mmproj_files[base] = {"path": full, "size_gb": size_gb, "name": f}
    
    # 第二次掃描：收集模型 + 對應 mmproj
    for root_dir, _, files in os.walk(MODEL_DIR):
        for f in files:
            if not f.endswith(".gguf") or "mmproj" in f.lower():
                continue
            full = os.path.join(root_dir, f)
            size_gb = os.path.getsize(full) / (1024**3)
            
            # 從檔名解析資訊
            name = f.replace(".gguf", "")
            parts = name.split("-")
            
            arch = "Unknown"
            quant = "Unknown"
            ctx_max = "Unknown"
            
            # 簡單解析
            for p in parts:
                if p.upper() in ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "F32", "IQ2", "IQM"):
                    quant = p.upper()
                if "MOE" in p.upper():
                    arch = "MoE"
                if "MOE" not in arch and "moe" in f.lower():
                    arch = "MoE"
                if p.startswith(("256K", "128K", "64K", "32K")):
                    ctx_max = p
            
            # 找對應的 mmproj
            found_mmproj = None
            for mmproj_base, mmproj_info in mmproj_files.items():
                if mmproj_base in name or name in mmproj_base:
                    found_mmproj = mmproj_info
                    break
            
            models.append({
                "path": full,
                "name": f,
                "size_gb": size_gb,
                "arch": arch,
                "quant": quant,
                "ctx_max": ctx_max,
                "dir": os.path.basename(root_dir),
                "mmproj": found_mmproj,  # 可能為 None
            })
    
    return models, mmproj_map


# ==================== API 狀態檢查 ====================
def check_api_status(port=8080):
    """檢查 API 整合狀態"""
    status = {"llama_server": "unchecked", "litellm": "unchecked", "openwebui": "unchecked"}
    
    # llama.cpp server
    try:
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        resp = urllib.request.urlopen(req, timeout=3)
        if resp.status == 200:
            status["llama_server"] = "available"
        else:
            status["llama_server"] = "error"
    except Exception:
        status["llama_server"] = "offline"
    
    # LiteLLM (預設 port 4000)
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:4000/health")
        resp = urllib.request.urlopen(req, timeout=3)
        if resp.status == 200:
            status["litellm"] = "available"
        else:
            status["litellm"] = "error"
    except Exception:
        status["litellm"] = "offline"
    
    # OpenWebUI (預設 port 3000)
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:3000/api/health")
        resp = urllib.request.urlopen(req, timeout=3)
        if resp.status == 200:
            status["openwebui"] = "available"
        else:
            status["openwebui"] = "error"
    except Exception:
        status["openwebui"] = "offline"
    
    return status


# ==================== 使用模式快速選擇 ====================
MODE_PRESETS = {
    "🗣️ 日常聊天": {
        "ctx": "32768", "moe_cpu": "32", "ngl": "99", "fa": True, "mmap": False,
        "kt": "q4_0", "vt": "q4_0", "desc": "32K 上下文，快速回應。適合一般對話。"
    },
    "💻 Coding 輔助": {
        "ctx": "65536", "moe_cpu": "36", "ngl": "99", "fa": True, "mmap": False,
        "kt": "q4_0", "vt": "q4_0", "desc": "64K 上下文，足夠讀取整個檔案。適合程式碼生成。"
    },
    "📄 長文分析": {
        "ctx": "131072", "moe_cpu": "34", "ngl": "99", "fa": True, "mmap": False,
        "kt": "q4_0", "vt": "q4_0", "desc": "128K 上下文。適合文件分析、摘要。"
    },
    "🔗 RAG 檢索": {
        "ctx": "196608", "moe_cpu": "33", "ngl": "99", "fa": True, "mmap": False,
        "kt": "q4_0", "vt": "q4_0", "desc": "192K 上下文。適合 RAG 系統，保留更多 RAM 給向量資料庫。"
    },
    "🚀 極限 256K": {
        "ctx": "262144", "moe_cpu": "32", "ngl": "99", "fa": True, "mmap": False,
        "kt": "q4_0", "vt": "q4_0", "desc": "256K 上下文，本機實測最佳配置。RTX 4070 12GB VRAM 65% 使用率。"
    },
    "⚡ Benchmark": {
        "ctx": "8192", "moe_cpu": "24", "ngl": "99", "fa": True, "mmap": False,
        "kt": "q4_0", "vt": "q4_0", "desc": "最小上下文 + 最多 MoE 上 GPU。速度測試用。"
    },
}


class LlamaLauncherApp:
    """llama.cpp Expert Launcher 主視窗"""

    def __init__(self, root):
        self.root = root
        self.root.title("🦙 llama.cpp Expert Launcher")
        self.root.geometry("880x760")
        self.root.minsize(800, 700)
        self.server_process = None
        
        # 硬體資訊
        self.hw = detect_hardware()
        # 收藏參數
        self.favorites = load_favorites()
        # 模型庫
        self.models, self.mmproj_map = scan_models_detailed()
        # API 狀態
        self.api_status = check_api_status(DEFAULT_PORT)

        # 資源預估變數
        self.est_vram = tk.StringVar(value="0.0")
        self.est_ram = tk.StringVar(value="0.0")
        self.est_kv = tk.StringVar(value="0.0")
        self.est_status = tk.StringVar(value="🟡 就緒")
        self.est_status_color = tk.StringVar(value="orange")

        # 樣式
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Microsoft JhengHei", 10, "bold"))
        style.configure("TLabelframe.Label", font=("Microsoft JhengHei", 10, "bold"))
        style.configure("HW.TLabel", font=("Microsoft JhengHei UI", 9))
        style.configure("Status.TLabel", font=("Microsoft JhengHei UI", 9))

        self.mmproj_path = tk.StringVar()
        self.var_mmproj = tk.BooleanVar(value=False)
        self.var_mmproj_offload = tk.BooleanVar(value=True)
        self.model_path = tk.StringVar()
        self.create_widgets()
        self.update_resource_estimate()
        
        # 自動套用最佳預設
        self.cb_mode.current(4)  # 極限 256K
        self.on_mode_change(None)
        
        # 初始狀態偵測
        self.root.after(500, self.refresh_all_status)

    # ---------- UI 建構 ----------
    def create_widgets(self):
        # === 0. 硬體資訊區 ===
        f0 = ttk.LabelFrame(self.root, text=" 🖥️ 系統硬體資訊 (自動偵測) ", padding=6)
        f0.pack(fill="x", padx=12, pady=6)
        
        hw_frame = ttk.Frame(f0)
        hw_frame.pack(fill="x")
        
        hw_items = [
            (f"CPU\n{self.hw['cpu']}", "w"),
            (f"RAM\n{self.hw['ram_gb']} GB", "center"),
            (f"GPU\n{self.hw['gpu']}", "center"),
            (f"VRAM\n{self.hw['vram_gb']} GB", "center"),
            (f"CUDA\n{self.hw['cuda']}", "center"),
            (f"llama.cpp\nb9060", "center"),
        ]
        for i, (text, anchor) in enumerate(hw_items):
            lbl = ttk.Label(hw_frame, text=text, relief="groove", padding=4, width=16, anchor=anchor)
            lbl.grid(row=0, column=i, padx=3, sticky="ew")
            lbl.configure(font=("Consolas", 9))
        
        # === 1. 模型選擇 ===
        f1 = ttk.LabelFrame(self.root, text=" 1. 選擇 GGUF 模型檔案 ", padding=8)
        f1.pack(fill="x", padx=12, pady=4)

        # 建立模型選擇子列 Frame，防止 pack 混雜排版擠壓
        f1_model = ttk.Frame(f1)
        f1_model.pack(fill="x", expand=True, pady=2)

        self.cb_model = ttk.Combobox(f1_model, textvariable=self.model_path, state="readonly", width=70)
        self.cb_model.pack(side="left", padx=4, fill="x", expand=True)
        self.cb_model.bind("<<ComboboxSelected>>", lambda e: self.on_model_change())

        ttk.Button(f1_model, text="📁 瀏覽", command=self.browse_model, width=10).pack(side="left", padx=2)
        ttk.Button(f1_model, text="🔄 掃描", command=self.refresh_models, width=10).pack(side="left", padx=2)

        # 模型資訊列 (獨立一行，錨定左側，擴展寬度防切)
        self.lbl_model_info = ttk.Label(f1, text="未選擇模型", foreground="gray", wraplength=800, font=("Microsoft JhengHei UI", 9))
        self.lbl_model_info.pack(fill="x", anchor="w", padx=6, pady=2)
        
        # mmproj 多模態選擇器（獨立一行）
        f1_mm = ttk.Frame(f1)
        f1_mm.pack(fill="x", pady=2)
        # 勾選框放左邊
        ttk.Checkbutton(f1_mm, text="👁️ 多模態", variable=self.var_mmproj, command=self.on_mmproj_change).pack(side="left", padx=4)
        # Combobox 放中間
        self.cb_mmproj = ttk.Combobox(f1_mm, textvariable=self.mmproj_path, state="readonly", width=40)
        self.cb_mmproj.pack(side="left", padx=4, fill="x", expand=True)
        self.cb_mmproj.bind("<<ComboboxSelected>>", lambda e: self.on_mmproj_change())
        # 資訊放右邊 (排在 Combobox 後方，防右側邊界切除)
        self.lbl_mmproj_info = ttk.Label(f1_mm, text="", foreground="green", font=("Microsoft JhengHei UI", 8))
        self.lbl_mmproj_info.pack(side="left", padx=8)

        # === 2. 使用模式快速選擇 ===
        f2 = ttk.LabelFrame(self.root, text=" 2. 使用模式 (一鍵套用) ", padding=6)
        f2.pack(fill="x", padx=12, pady=4)
        
        self.cb_mode = ttk.Combobox(f2, values=list(MODE_PRESETS.keys()), state="readonly", width=55)
        self.cb_mode.pack(side="left", padx=4)
        self.cb_mode.bind("<<ComboboxSelected>>", self.on_mode_change)

        # 收藏按鈕區
        fav_frame = ttk.Frame(f2)
        fav_frame.pack(side="right", padx=4)
        
        ttk.Button(fav_frame, text="⭐ 儲存為收藏", command=self.save_favorite_dialog, width=14).pack(side="left", padx=2)
        
        self.cb_fav = ttk.Combobox(fav_frame, values=[], state="readonly", width=15)
        self.cb_fav.pack(side="left", padx=2)
        self.cb_fav.bind("<<ComboboxSelected>>", self.on_favorite_select)
        self.refresh_favorites()

        self.lbl_mode_desc = ttk.Label(f2, text="", foreground="blue", wraplength=600)
        self.lbl_mode_desc.pack(fill="x", padx=6, pady=2)

        # === 3. VRAM / RAM 資源預估 ===
        f3 = ttk.LabelFrame(self.root, text=" 📊 資源預估 (即時) ", padding=6)
        f3.pack(fill="x", padx=12, pady=4)
        
        est_frame = ttk.Frame(f3)
        est_frame.pack(fill="x")
        
        # 使用兩行顯示，避免被截斷
        # 第一行：VRAM + 狀態
        fr_top = ttk.Frame(est_frame)
        fr_top.pack(fill="x", pady=2)
        
        self.lbl_est_vram = ttk.Label(fr_top, text="VRAM: 0.0 GB", foreground="green", font=("Consolas", 11, "bold"))
        self.lbl_est_vram.pack(side="left", padx=8)
        
        self.lbl_est_status = ttk.Label(fr_top, text="🟡 就緒", 
                                         foreground="orange", 
                                         font=("Microsoft JhengHei UI", 11, "bold"))
        self.lbl_est_status.pack(side="right", padx=8)
        
        # 第二行：RAM + KV Cache + 剩餘
        fr_bot = ttk.Frame(est_frame)
        fr_bot.pack(fill="x", pady=2)
        
        self.lbl_est_ram = ttk.Label(fr_bot, text="RAM: 0.0 GB", foreground="green", font=("Consolas", 10, "bold"))
        self.lbl_est_ram.pack(side="left", padx=8)
        
        self.lbl_est_kv = ttk.Label(fr_bot, text="KV: 0.0 GB", foreground="blue", font=("Consolas", 10))
        self.lbl_est_kv.pack(side="left", padx=8)
        
        self.lbl_est_free = ttk.Label(fr_bot, text="剩餘: 0.0 GB", 
                                       foreground="green", font=("Consolas", 10))
        self.lbl_est_free.pack(side="right", padx=8)

        # 第三行：執行狀態 + API
        fr_row3 = ttk.Frame(est_frame)
        fr_row3.pack(fill="x", pady=1)

        self.lbl_model_running = ttk.Label(fr_row3, text="", foreground="gray", font=("Microsoft JhengHei UI", 9))
        self.lbl_model_running.pack(side="left", padx=8)

        self.lbl_port_status = ttk.Label(fr_row3, text="", foreground="gray", font=("Microsoft JhengHei UI", 9))
        self.lbl_port_status.pack(side="left", padx=8)

        self.lbl_api_status = ttk.Label(fr_row3, text="", foreground="gray", font=("Microsoft JhengHei UI", 9))
        self.lbl_api_status.pack(side="left", padx=8)

        self.lbl_ram_usage = ttk.Label(fr_row3, text="RAM: 待偵測", foreground="gray", font=("Microsoft JhengHei UI", 9))
        self.lbl_ram_usage.pack(side="left", padx=8)

        ttk.Button(fr_row3, text="🔄 刷新狀態", command=self.refresh_all_status, width=12).pack(side="right", padx=4)

        # === 4. 參數微調 ===
        f4 = ttk.LabelFrame(self.root, text=" 4. 參數微調 (附中文說明) ", padding=8)
        f4.pack(fill="x", padx=12, pady=4)
        
        # 建立左右雙欄
        lf = ttk.Frame(f4)
        lf.pack(side="left", fill="both", expand=True, padx=8)
        lf.columnconfigure(1, weight=1)
        
        rf = ttk.Frame(f4)
        rf.pack(side="right", fill="both", expand=True, padx=8)
        rf.columnconfigure(1, weight=1)
        
        # --- 左欄參數 (lf) ---
        l_row = 0
        
        # Context (改為 Combobox 支援下拉選取與手動輸入)
        ttk.Label(lf, text="上下文長度 (-c):").grid(row=l_row, column=0, sticky="w", pady=4)
        self.ent_ctx = ttk.Combobox(lf, values=["8192", "16384", "32768", "65536", "131072", "196608", "262144"], width=10)
        self.ent_ctx.grid(row=l_row, column=1, sticky="w", padx=6)
        self.ent_ctx.bind("<KeyRelease>", lambda e: self.on_param_change())
        self.ent_ctx.bind("<<ComboboxSelected>>", lambda e: self.on_param_change())
        ttk.Label(lf, text="💡 262144=256K。每 32K +180MB VRAM", foreground="blue", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=l_row, column=2, sticky="w")
        l_row += 1

        # MoE CPU 層 (改為 Spinbox 支援 +1/-1 微調)
        ttk.Label(lf, text="MoE CPU 層數 (--n-cpu-moe):").grid(row=l_row, column=0, sticky="w", pady=4)
        self.ent_moe = ttk.Spinbox(lf, from_=0, to=100, increment=1, width=8, command=self.on_param_change)
        self.ent_moe.grid(row=l_row, column=1, sticky="w", padx=6)
        self.ent_moe.bind("<KeyRelease>", lambda e: self.on_param_change())
        ttk.Label(lf, text="💡 小=省 RAM 吃 VRAM", foreground="blue", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=l_row, column=2, sticky="w")
        l_row += 1

        # NGL (改為 Spinbox 支援 +1/-1 微調)
        ttk.Label(lf, text="GPU 層數 (-ngl):").grid(row=l_row, column=0, sticky="w", pady=4)
        self.ent_ngl = ttk.Spinbox(lf, from_=0, to=200, increment=1, width=8, command=self.on_param_change)
        self.ent_ngl.grid(row=l_row, column=1, sticky="w", padx=6)
        self.ent_ngl.bind("<KeyRelease>", lambda e: self.on_param_change())
        ttk.Label(lf, text="💡 99=全上 GPU", foreground="blue", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=l_row, column=2, sticky="w")
        l_row += 1

        # CPU 推論執行緒 (-t) (改為 Spinbox 支援 +1/-1 微調)
        ttk.Label(lf, text="CPU 執行緒 (-t):").grid(row=l_row, column=0, sticky="w", pady=4)
        self.ent_threads = ttk.Spinbox(lf, from_=1, to=128, increment=1, width=8, command=self.on_param_change)
        self.ent_threads.insert(0, str(self.hw.get("physical_cores", 8)))
        self.ent_threads.grid(row=l_row, column=1, sticky="w", padx=6)
        self.ent_threads.bind("<KeyRelease>", lambda e: self.on_param_change())
        ttk.Label(lf, text=f"💡 建議為實體核心數（已偵測為 {self.hw.get('physical_cores', 8)}）", foreground="blue", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=l_row, column=2, sticky="w")
        l_row += 1

        # CPU 批次處理執行緒 (-tb) (改為 Spinbox 支援 +1/-1 微調)
        ttk.Label(lf, text="批次執行緒 (-tb):").grid(row=l_row, column=0, sticky="w", pady=4)
        self.ent_threads_batch = ttk.Spinbox(lf, from_=1, to=128, increment=1, width=8, command=self.on_param_change)
        self.ent_threads_batch.insert(0, str(self.hw.get("physical_cores", 8)))
        self.ent_threads_batch.grid(row=l_row, column=1, sticky="w", padx=6)
        self.ent_threads_batch.bind("<KeyRelease>", lambda e: self.on_param_change())
        ttk.Label(lf, text="💡 Prompt 處理，建議與 -t 相同", foreground="blue", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=l_row, column=2, sticky="w")
        l_row += 1

        # Port
        ttk.Label(lf, text="埠號 (--port):").grid(row=l_row, column=0, sticky="w", pady=4)
        self.ent_port = ttk.Entry(lf, width=10)
        self.ent_port.insert(0, str(DEFAULT_PORT))
        self.ent_port.grid(row=l_row, column=1, sticky="w", padx=6)
        ttk.Label(lf, text="💡 預設 8080", foreground="blue", font=("Microsoft JhengHei UI", 8)).grid(row=l_row, column=2, sticky="w")
        
        # --- 右欄參數 (rf) ---
        r_row = 0
        
        # KV Cache K
        ttk.Label(rf, text="K 快取 (--cache-type-k):").grid(row=r_row, column=0, sticky="w", pady=4)
        self.cb_k = ttk.Combobox(rf, values=["f16", "q8_0", "q4_0"], state="readonly", width=8)
        self.cb_k.grid(row=r_row, column=1, sticky="w", padx=6)
        self.cb_k.bind("<<ComboboxSelected>>", lambda e: self.on_param_change())
        ttk.Label(rf, text="💡 q4_0 省 VRAM", foreground="blue", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=r_row, column=2, sticky="w")
        r_row += 1

        # KV Cache V
        ttk.Label(rf, text="V 快取 (--cache-type-v):").grid(row=r_row, column=0, sticky="w", pady=4)
        self.cb_v = ttk.Combobox(rf, values=["f16", "q8_0", "q4_0"], state="readonly", width=8)
        self.cb_v.grid(row=r_row, column=1, sticky="w", padx=6)
        self.cb_v.bind("<<ComboboxSelected>>", lambda e: self.on_param_change())
        ttk.Label(rf, text="💡 同 K 快取設定", foreground="blue", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=r_row, column=2, sticky="w")
        r_row += 1

        # Flash Attention
        self.var_fa = tk.BooleanVar(value=True)
        cb_fa = ttk.Checkbutton(rf, text="Flash Attention (--flash-attn)", variable=self.var_fa, command=self.on_param_change)
        cb_fa.grid(row=r_row, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(rf, text="💡 必開！加速 + 省 VRAM", foreground="green", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=r_row, column=2, sticky="w")
        r_row += 1

        # No MMAP
        self.var_mmap = tk.BooleanVar(value=False)
        cb_mmap = ttk.Checkbutton(rf, text="mmap (取消勾選 = --no-mmap)", variable=self.var_mmap, command=self.on_param_change)
        cb_mmap.grid(row=r_row, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(rf, text="💡 RAM 緊時取消 = 防止 swap 卡頓", foreground="green", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=r_row, column=2, sticky="w")
        r_row += 1

        # Reasoning
        self.var_reason = tk.BooleanVar(value=False)
        cb_reason = ttk.Checkbutton(rf, text="Reasoning 模式 (不勾 = --reasoning off)", variable=self.var_reason, command=self.on_param_change)
        cb_reason.grid(row=r_row, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(rf, text="💡 一般對話關閉，避免回傳空白 content", foreground="green", font=("Microsoft JhengHei UI", 8), wraplength=220).grid(row=r_row, column=2, sticky="w")

        # === 5. 指令預覽 ===
        f5 = ttk.LabelFrame(self.root, text=" 5. 指令預覽 ", padding=6)
        f5.pack(fill="x", padx=12, pady=4)

        self.txt_cmd = tk.Text(f5, height=5, width=85, wrap="char", font=("Consolas", 9))
        self.txt_cmd.pack(fill="x", pady=2)

        # === 6. 控制按鈕 ===
        f7 = ttk.Frame(self.root)
        f7.pack(fill="x", padx=12, pady=8)

        ttk.Button(f7, text="🔄 刷新預估", command=self.update_resource_estimate).pack(side="left", padx=4)
        self.btn_run = ttk.Button(f7, text="🚀 啟動伺服器", command=self.run_server, style="Accent.TButton")
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(f7, text="🛑 停止伺服器", command=self.stop_server, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(f7, text="✂ 複製指令", command=self.copy_cmd).pack(side="left", padx=4)
        ttk.Button(f7, text="🧪 健康檢查", command=self.health_check).pack(side="left", padx=4)

        # 狀態列
        self.lbl_status = ttk.Label(self.root, text="🟡 就緒 | 等待啟動", foreground="gray")
        self.lbl_status.pack(fill="x", padx=12, pady=2)

    # ---------- 模型掃描 ----------
    def scan_models(self):
        """掃描 MODEL_DIR 下所有 .gguf 檔案"""
        self.models, self.mmproj_map = scan_models_detailed()
        models_display = []
        for m in self.models:
            display = f"{m['path']}  ({m['size_gb']:.1f} GB)"
            models_display.append(display)
        return models_display

    def refresh_models(self):
        """重新掃描模型"""
        models_display = self.scan_models()
        if models_display:
            self.cb_model["values"] = models_display
            self.cb_model.current(0)
            self.model_path.set(models_display[0].split("  (")[0])
            self.on_model_change()
        else:
            self.cb_model["values"] = ["未找到 GGUF 檔案"]
            self.lbl_model_info.config(text="未找到模型", foreground="red")

    def on_model_change(self):
        """模型切換時更新資訊"""
        path = self.model_path.get().split("  (")[0]
        # 找詳細資訊
        found_mmproj = None
        for m in self.models:
            if m["path"] == path:
                info = f"📦 {m['name']} | 大小: {m['size_gb']:.1f} GB | 架構: {m['arch']} | 量化: {m['quant']} | 目錄: {m['dir']}"
                self.lbl_model_info.config(text=info, foreground="blue")
                if m.get("mmproj"):
                    found_mmproj = m["mmproj"]
                break
        else:
            self.lbl_model_info.config(text="未選擇模型", foreground="gray")
        
        # 自動偵測並推薦 mmproj
        if found_mmproj:
            self.cb_mmproj["values"] = [found_mmproj["name"]]
            self.cb_mmproj.set(found_mmproj["name"])
            self.lbl_mmproj_info.config(
                text=f"📷 找到 {found_mmproj['name']} ({found_mmproj['size_gb']:.1f}GB)",
                foreground="green"
            )
            self.var_mmproj.set(True)  # 自動啟用
        else:
            self.cb_mmproj["values"] = []
            self.cb_mmproj.set("")
            self.lbl_mmproj_info.config(text="無 mmproj", foreground="gray")
            self.var_mmproj.set(False)
        
        self.update_resource_estimate()

    def on_mmproj_change(self):
        """mmproj 切換時更新"""
        self.update_cmd()
        self.update_resource_estimate()

    # ---------- 狀態偵測 ----------
    def refresh_all_status(self):
        """刷新所有狀態"""
        port = int(self.ent_port.get() or DEFAULT_PORT)
        
        # 檢查 llama-server 是否執行中
        running_text = "🔴 無"
        running_color = "red"
        port_text = f"Port:{port}"
        port_color = "gray"
        
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq llama-server.exe"],
                capture_output=True, text=True, timeout=5
            )
            if "llama-server.exe" in result.stdout:
                try:
                    import urllib.request
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                    resp = urllib.request.urlopen(req, timeout=2)
                    if resp.status == 200:
                        running_text = "🟢 執行中"
                        running_color = "green"
                        port_text = f"Port:{port}"
                        port_color = "green"
                    else:
                        running_text = "🟡 異常"
                        running_color = "orange"
                except Exception:
                    running_text = "🟡 無回應"
                    running_color = "orange"
        except Exception:
            running_text = "❌ 錯誤"
        
        self.lbl_model_running.config(text=f"模型: {running_text}", foreground=running_color)
        self.lbl_port_status.config(text=port_text, foreground=port_color)
        
        # API 狀態
        api_parts = []
        for key, name, url in [("llama_server", f"llama:{port}", f"http://127.0.0.1:{port}"),
                                 ("litellm", "LiteLLM", "http://127.0.0.1:4000"),
                                 ("openwebui", "WebUI", "http://127.0.0.1:3000")]:
            try:
                import urllib.request
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health" if key == "llama_server" else url)
                urllib.request.urlopen(req, timeout=1)
                api_parts.append(f"🟢 {name}")
            except Exception:
                api_parts.append(f"⚪ {name}")
        self.lbl_api_status.config(text="API: " + " ".join(api_parts), foreground="green" if any("🟢" in p for p in api_parts) else "gray")
        
        # 系統 RAM
        try:
            import psutil
            ram = psutil.virtual_memory()
            used_gb = ram.used / (1024**3)
            total_gb = ram.total / (1024**3)
            pct = ram.percent
            color = "green" if pct < 70 else "orange" if pct < 90 else "red"
            self.lbl_ram_usage.config(
                text=f"RAM: {used_gb:.1f}/{total_gb:.1f}GB ({pct:.0f}%)",
                foreground=color
            )
        except ImportError:
            self.lbl_ram_usage.config(text="RAM: 需安裝 psutil", foreground="gray")
        except Exception:
            self.lbl_ram_usage.config(text="RAM: 無法偵測", foreground="gray")

    def check_port_busy(self, port):
        """檢查 port 是否被佔用"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="選擇 GGUF 模型",
            filetypes=[("GGUF 模型", "*.gguf"), ("所有檔案", "*.*")],
            initialdir=MODEL_DIR,
        )
        if path:
            self.model_path.set(path)
            self.on_model_change()

    # ---------- 使用模式切換 ----------
    def on_mode_change(self, event):
        """切換使用模式"""
        name = self.cb_mode.get()
        p = MODE_PRESETS.get(name)
        if not p:
            return
        
        self.lbl_mode_desc.config(text=f"📋 {p['desc']}")
        
        # 套用參數
        self.ent_ctx.delete(0, tk.END)
        self.ent_ctx.insert(0, p["ctx"])
        self.ent_moe.delete(0, tk.END)
        self.ent_moe.insert(0, p["moe_cpu"])
        self.ent_ngl.delete(0, tk.END)
        self.ent_ngl.insert(0, p["ngl"])
        
        # 套用執行緒 (優先使用預設核心數)
        threads_val = p.get("threads", str(self.hw.get("physical_cores", 8)))
        self.ent_threads.delete(0, tk.END)
        self.ent_threads.insert(0, threads_val)
        
        threads_batch_val = p.get("threads_batch", str(self.hw.get("physical_cores", 8)))
        self.ent_threads_batch.delete(0, tk.END)
        self.ent_threads_batch.insert(0, threads_batch_val)
        
        self.cb_k.set(p["kt"])
        self.cb_v.set(p["vt"])
        self.var_fa.set(p["fa"])
        self.var_mmap.set(p["mmap"])
        
        self.update_cmd()
        self.update_resource_estimate()

    # ---------- 收藏參數 ----------
    def refresh_favorites(self):
        """刷新收藏列表"""
        self.favorites = load_favorites()
        self.cb_fav["values"] = [f["name"] for f in self.favorites]

    def on_favorite_select(self, event):
        """套用收藏參數"""
        name = self.cb_fav.get()
        for fav in self.favorites:
            if fav["name"] == name:
                self.ent_ctx.delete(0, tk.END)
                self.ent_ctx.insert(0, str(fav.get("ctx", 8192)))
                self.ent_moe.delete(0, tk.END)
                self.ent_moe.insert(0, str(fav.get("moe_cpu", 32)))
                self.ent_ngl.delete(0, tk.END)
                self.ent_ngl.insert(0, str(fav.get("ngl", 99)))
                
                # 載入收藏的執行緒參數
                self.ent_threads.delete(0, tk.END)
                self.ent_threads.insert(0, str(fav.get("threads", self.hw.get("physical_cores", 8))))
                self.ent_threads_batch.delete(0, tk.END)
                self.ent_threads_batch.insert(0, str(fav.get("threads_batch", self.hw.get("physical_cores", 8))))
                
                self.cb_k.set(fav.get("kt", "q4_0"))
                self.cb_v.set(fav.get("vt", "q4_0"))
                self.var_fa.set(fav.get("fa", True))
                self.var_mmap.set(fav.get("mmap", False))
                self.lbl_mode_desc.config(text=f"⭐ 已套用收藏: {name}")
                self.update_cmd()
                self.update_resource_estimate()
                break

    def save_favorite_dialog(self):
        """儲存收藏參數對話框"""
        win = tk.Toplevel(self.root)
        win.title("⭐ 儲存收藏參數")
        win.geometry("350x200")
        win.transient(self.root)
        win.grab_set()
        
        ttk.Label(win, text="名稱:").pack(pady=4)
        ent = ttk.Entry(win, width=30)
        ent.pack(pady=4)
        ent.insert(0, f"我的 {self.ent_ctx.get()} 配置")
        
        def save():
            name = ent.get().strip()
            if not name:
                messagebox.showwarning("⚠️", "名稱不能空白")
                return
            params = {
                "ctx": self.ent_ctx.get(),
                "moe_cpu": self.ent_moe.get(),
                "ngl": self.ent_ngl.get(),
                "threads": self.ent_threads.get(),
                "threads_batch": self.ent_threads_batch.get(),
                "kt": self.cb_k.get(),
                "vt": self.cb_v.get(),
                "fa": self.var_fa.get(),
                "mmap": self.var_mmap.get(),
            }
            self.favorites = save_favorite(name, params)
            self.refresh_favorites()
            messagebox.showinfo("✅", f"已儲存收藏: {name}")
            win.destroy()
        
        ttk.Button(win, text="儲存", command=save).pack(pady=8)
        ttk.Button(win, text="取消", command=win.destroy).pack()

    # ---------- 資源預估 ----------
    def update_resource_estimate(self):
        """更新資源預估顯示"""
        try:
            ctx = int(self.ent_ctx.get() or 8192)
            moe = int(self.ent_moe.get() or 32)
            ngl = int(self.ent_ngl.get() or 99)
            kt = self.cb_k.get() or "q4_0"
            vt = self.cb_v.get() or "q4_0"
        except ValueError:
            return
        
        # 取得 mmproj 大小 (BF16 約一半上 VRAM)
        mmproj_gb = 0
        if self.var_mmproj.get() and self.mmproj_path.get():
            for m in self.models:
                if m.get("mmproj") and m["mmproj"]["name"] == self.mmproj_path.get():
                    mmproj_gb = m["mmproj"]["size_gb"] * 0.5  # BF16 僅一半上 GPU
                    break
        
        est = estimate_resources(ctx, moe, ngl, kt, vt, self.hw["vram_gb"], self.hw["ram_gb"], mmproj_gb)
        
        self.est_vram.set(str(est["vram_gb"]))
        self.est_ram.set(str(est["ram_gb"]))
        self.est_kv.set(str(est["kv_vram_gb"]))
        self.est_status.set(est["status"])
        self.est_status_color.set(est["status_color"])
        
        self.lbl_est_vram.config(text=f"VRAM: {est['vram_gb']} GB / {self.hw['vram_gb']} GB")
        self.lbl_est_vram.config(foreground=est["status_color"])
        
        self.lbl_est_ram.config(text=f"RAM: {est['ram_gb']} GB / {self.hw['ram_gb']} GB")
        
        self.lbl_est_kv.config(text=f"KV Cache: {est['kv_vram_gb']} GB")
        
        self.lbl_est_status.config(text=est["status"], foreground=est["status_color"])
        
        free_vram = self.hw["vram_gb"] - est["vram_gb"]
        free_color = "green" if free_vram > 1.0 else "red" if free_vram < 0 else "orange"
        self.lbl_est_free.config(
            text=f"剩餘 VRAM: {free_vram:.1f} GB",
            foreground=free_color
        )

    def on_param_change(self):
        """參數變更時更新"""
        self.update_cmd()
        self.update_resource_estimate()

    # ---------- 健康檢查 ----------
    def health_check(self):
        """執行健康檢查"""
        port = int(self.ent_port.get() or DEFAULT_PORT)
        status = check_api_status(port)
        
        lines = ["=== 健康檢查結果 ===\n"]
        for key, name, url in [("llama_server", "llama.cpp", f"http://127.0.0.1:{port}"),
                                 ("litellm", "LiteLLM", "http://127.0.0.1:4000"),
                                 ("openwebui", "OpenWebUI", "http://127.0.0.1:3000")]:
            s = status.get(key, "unchecked")
            icon = "✅" if s == "available" else "❌" if s == "error" else "⬜"
            lines.append(f"{icon} {name}: {s} ({url})")
        
        lines.append(f"\n=== 硬體資訊 ===")
        lines.append(f"CPU: {self.hw['cpu']}")
        lines.append(f"RAM: {self.hw['ram_gb']} GB")
        lines.append(f"GPU: {self.hw['gpu']} ({self.hw['vram_gb']} GB)")
        lines.append(f"CUDA: {self.hw['cuda']}")
        
        messagebox.showinfo("健康檢查", "\n".join(lines))

    # ---------- 指令建構 ----------
    def build_command(self):
        model = self.model_path.get().split("  (")[0]  # 去除大小標籤
        if not model or not os.path.isfile(model):
            return "# 請先選擇有效的 GGUF 模型"

        parts = [
            f'"{LLAMA_SERVER}"',
            f'-m "{model}"',
            f"-c {self.ent_ctx.get() or 8192}",
            f"-ngl {self.ent_ngl.get() or 99}",
            f"-t {self.ent_threads.get() or 8}",
            f"-tb {self.ent_threads_batch.get() or 8}",
            f"--n-cpu-moe {self.ent_moe.get() or 32}",
            f"--cache-type-k {self.cb_k.get() or 'q4_0'}",
            f"--cache-type-v {self.cb_v.get() or 'q4_0'}",
            f"--host 127.0.0.1",
            f"--port {self.ent_port.get() or DEFAULT_PORT}",
        ]
        
        # mmproj 多模態支援
        if self.var_mmproj.get() and self.mmproj_path.get():
            # 從 models 中找完整路徑
            mmproj_full = None
            for m in self.models:
                if m.get("mmproj") and m["mmproj"]["name"] == self.mmproj_path.get():
                    mmproj_full = m["mmproj"]["path"]
                    break
            if mmproj_full:
                parts.append(f'-mm "{mmproj_full}"')
                if self.var_mmproj_offload.get():
                    parts.append("--mmproj-offload")
                else:
                    parts.append("--no-mmproj-offload")
        
        if self.var_fa.get():
            parts.append("--flash-attn on")
        else:
            parts.append("--flash-attn off")
        if not self.var_mmap.get():
            parts.append("--no-mmap")
        if not self.var_reason.get():
            parts.append("--reasoning off")
        return " ^\n  ".join(parts)

    def update_cmd(self):
        self.txt_cmd.delete("1.0", tk.END)
        self.txt_cmd.insert("1.0", self.build_command())

    def copy_cmd(self):
        cmd = " ".join(self.build_command().split(" ^\n  "))
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.lbl_status.config(text="📋 指令已複製到剪貼簿", foreground="blue")

    # ---------- 伺服器控制 ----------
    def run_server(self):
        model = self.model_path.get().split("  (")[0]
        if not model or not os.path.isfile(model):
            messagebox.showwarning("⚠️ 警告", "請先選擇有效的 GGUF 模型檔案！")
            return

        port = int(self.ent_port.get() or DEFAULT_PORT)
        
        # 直接檢查 port 是否被佔用
        port_busy = False
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            port_busy = (result == 0)
        except Exception:
            pass
        
        if port_busy:
            reply = messagebox.askyesno(
                "⚠️ Port 已被佔用",
                f"Port {port} 已有程式在執行。\n\n"
                "[是] 停止現有並重啟\n"
                "[否] 取消"
            )
            if not reply:
                return
            # 先強制停止
            self.stop_server()
        
        self._start_server(port, model)

    def _start_server(self, port, model):
        """實際啟動伺服器"""
        cmd_line = " ".join(self.build_command().split(" ^\n  "))
        self.update_cmd()

        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            # 寫入 bat 檔再執行，避免 shell=True 引號問題
            bat_path = os.path.join(LOG_DIR, "_launcher_start.bat")
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write("@echo off\n")
                f.write(cmd_line + "\n")
                f.write("if %errorlevel% neq 0 pause\n")
            
            self.server_process = subprocess.Popen(
                bat_path,
                shell=True,
                cwd=WORK_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            self.btn_run.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.lbl_status.config(
                text=f"🟢 伺服器已啟動 | http://127.0.0.1:{port}",
                foreground="green",
            )
            self.refresh_all_status()
        except Exception as e:
            messagebox.showerror("❌ 錯誤", f"無法啟動: {e}")

    def stop_server(self):
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
        # 也嘗試殺掉殘留的 llama-server
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl_status.config(text="🔴 伺服器已停止", foreground="red")


# ==================== 入口 ====================
if __name__ == "__main__":
    # 檢查 llama-server 是否存在
    if not os.path.isfile(LLAMA_SERVER):
        print(f"❌ 找不到 llama-server.exe")
        print(f"   預期路徑: {LLAMA_SERVER}")
        print(f"   請確認 llama.cpp 已正確部署")
        sys.exit(1)

    root = tk.Tk()
    app = LlamaLauncherApp(root)
    # 設定視窗置中
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    root.mainloop()
