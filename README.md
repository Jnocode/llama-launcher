# 🦙 llama.cpp Expert Launcher (GUI)

為 Windows 與 NVIDIA CUDA 環境打造的高質感 `llama.cpp` 專家級啟動器。本工具專為 Mixture of Experts (MoE，混合專家) 模型（如 Qwen 35B MoE 等）以及長上下文 (Long Context) 進行效能校準與優化，提供直覺且精確的資源估算與參數微調介面。

---

## 🌟 核心特色 (Key Features)

### 1. 🧠 實體 CPU 核心智慧偵測
*   **痛點解決**：以往啟動器僅偵測邏輯處理器數，若將線程 `-t` 設為超線程（Hyper-Threading）後的虛擬核心，將導致 CPU 推理效能發生嚴重雪崩。
*   **優化方案**：底層透過 Windows `wmic` 繞過虛擬核心，**直接鎖定您的「實體 CPU 核心數」**，並自動填入為最優預設值，完美釋放 CPU 算力。
*   **靜音機制**：新版 Win11 停用 `wmic` 產生的 CLI 警告已被徹底靜音重導向，若無 wmic 將自動且安全地退回至邏輯核心折半的 Python 原生安全預設值。

### 2. 🎛️ 現代化「左右雙欄」排版 (Two-Column Layout)
*   **版面重構**：告別傳統單列拉長的擁擠 UI。將參數依功能性質劃分為左側數值欄（Context, MoE, NGL, Threads）與右側快取/開關欄（K/V Caches, FA, MMAP, Reasoning）。
*   **完美可讀性**：提示文字的折行寬度（`wraplength`）調寬，保證在中文化說明下 **100% 視覺無遮擋**，且主視窗比例完美收緊（`880x760`），按鈕再也不會被擠出畫面！

### 3. 📈 即時資源與 KV Cache 估算
*   基於實際硬體數據（顯存 VRAM、系統記憶體 DRAM）校準的資源預估公式。
*   即時計算並呈現在設定 Context 下的 **KV Cache 顯存開銷**，並以 `🟢 安全`、`🟡 危險`、`🔴 爆顯存` 等燈號實時給予調參反饋。

### 4. ⚡ 旋鈕與下拉選單微調 (Spinbox & Combobox)
*   **上下文長度 (-c)**：升級為混合式 `Combobox` 下拉選單，內建 `8K` 至 `256K` 等標準常用選項，同時保留手動輸入任意數字的自由度，點選後預估立即連動。
*   **數值欄位**：`MoE CPU 層數`、`GPU 層數 (-ngl)`、`CPU 執行緒 (-t)`、`批次執行緒 (-tb)` 全面引入 `ttk.Spinbox` (+1/-1 按鈕)，點擊微調箭頭即可毫秒級更新命令與資源預估。

### 5. ⭐ 模式預設與收藏夾
*   內建「日常聊天」、「Coding 輔助」、「長文分析」、「極限 256K」與「Benchmark」一鍵套用模式。
*   支援將自訂的調參（含執行緒、快取設定）一鍵儲存至收藏夾，下次開啟自動讀取。

### 6. 📤 卸載模型與顯存一鍵釋放 (Unload Model & VRAM Release)
*   **一鍵卸載**：新增「📤 卸載模型」按鈕，快速終止伺服器進程，不需關閉 GUI 即可 100% 釋放載入中模型所佔用的 VRAM 顯存與 RAM 記憶體。
*   **狀態同步**：按鈕狀態與伺服器執行狀態實時同步，確保僅在執行中可觸發卸載，且卸載後 UI 狀態與資源指標會同步刷新。

### 7. ⚙️ 設定持久化分頁 (Settings Persistence Tab)
*   **路徑預設值管理**：所有路徑欄位（Model Dir、Llama Dir、Log Dir）均附「瀏覽」按鈕，可一鍵選取資料夾。
*   **儲存/重置**：點選「💾 儲存預設」將所有路徑寫入 `launcher_settings.json`；點選「🔄 重置預設」清除自訂設定還原為程式內建值。

### 8. 🌐 WSL/LAN 網路模式 (Network Mode)
*   **跨 WSL 連線**：啟用後自動綁定 `0.0.0.0`，使 WSL 或其他區域網路裝置可透過 `host.docker.internal` 連線至本機 llama.cpp 服務。
*   **動態主機解析**：OpenClaw 設定自動使用 `host.docker.internal` 作為 baseUrl，不再需要手動修改配置。

### 9. 🔌 OpenClaw 端口自動補全 (Port Auto-Complete)
*   **開啟即偵測**：切換至 OpenClaw 設定分頁時，自動偵測 llama.cpp 目前使用的端口並填入配置。
*   **智能回退**：若無法自動偵測，保留原值不覆蓋，避免誤設定。

### 10. 🖼️ 多模態 mmproj 手動瀏覽強化 (Manual mmproj Browse)
*   **支援雙格式**：`.gguf` 與 `.bin` 格式皆可瀏覽選取，不再限單一格式。
*   **初始目錄優化**：優先定位至 `llama-server.exe` 所在目錄（mmproj 常見放置處）。
*   **切換模型不覆蓋**：手動選取 mmproj 後，即使掃描到新的 auto-detect 結果，系統保留手動選擇不受影響。

---

## 🚀 快速開始 (Quick Start)

### 1. 準備環境
本工具採用極其輕量的 `tkinter` 原生 GUI 架構，只需安裝系統偵測依賴 `psutil`：
```bash
# 1. 建立並啟用虛擬環境
python -m venv .venv
.venv\Scripts\activate

# 2. 安裝核心偵測依賴
pip install psutil
```

### 2. 執行啟動器
```bash
python artifacts/llama_launcher.py
```

---

## 🛠️ 技術規格與路徑配置
*   **預設 llama.cpp 伺服器路徑**：`D:\Workspace\artifacts\llama.cpp\b9060-cuda13.1\llama-server.exe`
*   **預設模型載入目錄**：`D:\Workspace\artifacts\models`
*   **配置檔 (收藏夾)**：`launcher_presets.json`

---

## 📄 授權條款
本專案採用 MIT 授權條款。
