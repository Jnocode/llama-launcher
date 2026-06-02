# Checkpoint: Llama Launcher - 實裝「卸載模型」功能 (v1.0.2 發佈完成)

## 已完成事項
- [x] 在 GUI 主介面 (`llama_launcher.py`) 控制面板新增「📤 卸載模型」按鈕。
- [x] 實作 `LlamaLauncherApp.unload_model(self)` 邏輯，綁定至 `stop_server()` 並在卸載後更新狀態列且重新偵測硬體 RAM/VRAM。
- [x] 在 `_start_server` 啟用 `self.btn_unload`，在 `stop_server` 停用 `self.btn_unload` 以同步狀態。
- [x] 撰寫並成功執行 `test_llama_launcher.py` 自動化單元測試，無介面 Tkinter 測試 100% 通過。
- [x] 產出 `walkthrough.md`。
- [x] 升級 `APP_VERSION` 版本號至 `"1.0.2"`。
- [x] 使用 PyInstaller 打包產生最新的 `dist/LlamaLauncher.exe` 執行檔。
- [x] 更新 `README.md` 與 `llama_launcher_README.md`，載入新功能說明。
- [x] 執行 Git 提交、建立 Release Tag `v1.0.2` 並成功推送至 GitHub 遠端倉庫。
- [x] 寫入跨平台全域記憶。

## 待辦事項
- [x] 開啟應用程式讓使用者直接進行操作驗證 (已啟動 `llama_launcher.py`)。

## 關鍵發現
- 在 `llama.cpp` 中，`llama-server.exe` 進程在執行時會實質鎖定 GGUF 模型的顯存與記憶體。因此，最乾淨且徹底的卸載模型做法是終止該伺服器進程。本實作在停止進程的同時，重設了按鈕狀態與顯存預估狀態，實現了完整的卸載效果。
- 透過 `pyinstaller LlamaLauncher.spec` 可以很方便地使用現有的 spec 配置一鍵編譯為獨立的單一執行檔，且打包路徑已在 `.gitignore` 中被正確排除。
