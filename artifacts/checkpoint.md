# Checkpoint: Llama Launcher - 實裝「卸載模型」功能

## 已完成事項
- [x] 在 GUI 主介面 (`llama_launcher.py`) 控制面板新增「📤 卸載模型」按鈕。
- [x] 實作 `LlamaLauncherApp.unload_model(self)` 邏輯，綁定至 `stop_server()` 並在卸載後更新狀態列且重新偵測硬體 RAM/VRAM。
- [x] 在 `_start_server` 啟用 `self.btn_unload`，在 `stop_server` 停用 `self.btn_unload` 以同步狀態。
- [x] 撰寫並成功執行 `test_llama_launcher.py` 自動化單元測試，無介面 Tkinter 測試 100% 通過。
- [x] 產出 `walkthrough.md`。

## 待辦事項
- [x] 寫入跨平台全域記憶與 Debug 日誌。
- [ ] 邀請使用者開啟專案主介面進行人工手動操作確認。

## 關鍵發現
- 在 `llama.cpp` 中，`llama-server.exe` 進程在執行時會實質鎖定 GGUF 模型的顯存與記憶體。因此，最乾淨且徹底的卸載模型做法是終止該伺服器進程。本實作在停止進程的同時，重設了按鈕狀態與顯存預估狀態，實現了完整的卸載效果。
