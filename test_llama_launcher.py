import unittest
import tkinter as tk
import os
import sys

# 將目前目錄加入 path 確保能 import 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_launcher import LlamaLauncherApp

class TestLlamaLauncherUnload(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # 隱藏視窗以避免在桌面上顯示
        self.app = LlamaLauncherApp(self.root)

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_unload_button_exists_and_disabled_by_default(self):
        """驗證卸載模型按鈕預設存在且為停用狀態"""
        self.assertTrue(hasattr(self.app, 'btn_unload'), "應包含 btn_unload 屬性")
        self.assertEqual(str(self.app.btn_unload['state']), 'disabled', "卸載按鈕預設應為停用狀態")

    def test_unload_model_sets_state_disabled_and_updates_status(self):
        """驗證呼叫 unload_model 後按鈕狀態更新與狀態列文字"""
        # 模擬啟動後的狀態
        self.app.btn_unload.config(state='normal')
        self.app.btn_stop.config(state='normal')
        
        # 執行卸載
        self.app.unload_model()
        
        self.assertEqual(str(self.app.btn_unload['state']), 'disabled', "卸載後卸載按鈕應停用")
        self.assertEqual(str(self.app.btn_stop['state']), 'disabled', "卸載後停止按鈕應停用")
        self.assertEqual(str(self.app.btn_run['state']), 'normal', "卸載後啟動按鈕應恢復可用")
        self.assertIn("模型已卸載", self.app.lbl_status['text'], "狀態列文字應包含「模型已卸載」")

if __name__ == '__main__':
    unittest.main()
