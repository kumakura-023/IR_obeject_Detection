# version_tracker.py
"""
共有バージョントラッカー - 全ファイルで同じ辞書を使用
このファイルを新規作成して、各ファイルからインポートして使う
"""

import datetime
import hashlib

# ★★★ 真のグローバル辞書（このモジュールで一元管理） ★★★
_GLOBAL_VERSION_TRACKERS = {}

class VersionTracker:
    """スクリプトのバージョンと修正履歴を追跡"""
    
    def __init__(self, script_name, version="1.0.0"):
        self.script_name = script_name
        self.version = version
        self.load_time = datetime.datetime.now()
        self.modifications = []
        
        # ★★★ 真のグローバル辞書に登録 ★★★
        _GLOBAL_VERSION_TRACKERS[script_name] = self
        
    def add_modification(self, description, author="AI Assistant"):
        """修正履歴を追加"""
        timestamp = datetime.datetime.now()
        self.modifications.append({
            'timestamp': timestamp,
            'description': description,
            'author': author
        })
        
    def get_file_hash(self, filepath):
        """ファイルのハッシュ値を計算（変更検出用）"""
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()[:8]
        except:
            return "unknown"
    
    def print_version_info(self):
        """バージョン情報を表示"""
        print(f"\n{'='*60}")
        print(f"📋 {self.script_name} - Version {self.version}")
        print(f"⏰ Loaded: {self.load_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if hasattr(self, 'file_hash'):
            print(f"🔗 File Hash: {self.file_hash}")
        
        if self.modifications:
            print(f"📝 Recent Modifications ({len(self.modifications)}):")
            for i, mod in enumerate(self.modifications[-3:], 1):  # 最新3件
                print(f"   {i}. {mod['timestamp'].strftime('%H:%M:%S')} - {mod['description']}")
        
        print(f"{'='*60}\n")

    @staticmethod
    def print_all_versions():
        """プロジェクト全体のバージョン情報を一括表示"""
        if not _GLOBAL_VERSION_TRACKERS:
            print("⚠️ バージョン管理対象のファイルが見つかりません")
            return
        
        print(f"\n{'='*80}")
        print(f"🚀 プロジェクト全体バージョン情報")
        print(f"⏰ 表示時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 管理対象ファイル数: {len(_GLOBAL_VERSION_TRACKERS)}")
        print(f"{'='*80}")
        
        # 読み込み時刻順にソート
        sorted_trackers = sorted(
            _GLOBAL_VERSION_TRACKERS.items(),
            key=lambda x: x[1].load_time
        )
        
        for i, (script_name, tracker) in enumerate(sorted_trackers, 1):
            print(f"\n{i}. 📄 {tracker.script_name}")
            print(f"   📌 Version: {tracker.version}")
            print(f"   ⏰ Loaded: {tracker.load_time.strftime('%H:%M:%S')}")
            
            if hasattr(tracker, 'file_hash') and tracker.file_hash:
                print(f"   🔗 Hash: {tracker.file_hash}")
            
            if tracker.modifications:
                latest_mod = tracker.modifications[-1]
                print(f"   📝 Latest: {latest_mod['timestamp'].strftime('%H:%M:%S')} - {latest_mod['description']}")
                if len(tracker.modifications) > 1:
                    print(f"   📋 Total modifications: {len(tracker.modifications)}")
            else:
                print(f"   📝 Modifications: None")
        
        print(f"\n{'='*80}")
        print(f"🎉 バージョン情報表示完了")
        print(f"{'='*80}\n")

    @staticmethod
    def print_version_summary():
        """コンパクトなバージョンサマリーを表示"""
        if not _GLOBAL_VERSION_TRACKERS:
            print("⚠️ バージョン管理対象のファイルが見つかりません")
            return
        
        print(f"\n📊 プロジェクトバージョンサマリー ({len(_GLOBAL_VERSION_TRACKERS)} files)")
        print("-" * 70)
        
        # 読み込み時刻順にソート
        sorted_trackers = sorted(
            _GLOBAL_VERSION_TRACKERS.items(),
            key=lambda x: x[1].load_time
        )
        
        for script_name, tracker in sorted_trackers:
            mod_count = len(tracker.modifications)
            latest_time = tracker.load_time.strftime('%H:%M:%S')
            print(f"📄 {tracker.script_name:<30} v{tracker.version:<8} ({mod_count} mods) {latest_time}")
        
        print("-" * 70)

    @staticmethod
    def get_all_trackers():
        """全てのトラッカーを取得（デバッグ用）"""
        return _GLOBAL_VERSION_TRACKERS.copy()

    @staticmethod
    def get_tracker_count():
        """登録されているトラッカー数を取得"""
        return len(_GLOBAL_VERSION_TRACKERS)


# 各ファイル用のバージョントラッカーを作成
def create_version_tracker(script_name, filepath=None):
    """バージョントラッカーを作成"""
    tracker = VersionTracker(script_name)
    
    if filepath:
        tracker.file_hash = tracker.get_file_hash(filepath)
    
    return tracker

# ===== 便利関数 =====
def show_all_project_versions():
    """プロジェクト全体のバージョン情報を表示（関数版）"""
    VersionTracker.print_all_versions()

def show_project_summary():
    """プロジェクトサマリーを表示（関数版）"""
    VersionTracker.print_version_summary()

def debug_version_status():
    """デバッグ用：現在の状況を確認"""
    trackers = VersionTracker.get_all_trackers()
    print(f"🔍 デバッグ情報:")
    print(f"   グローバル辞書のアドレス: {id(_GLOBAL_VERSION_TRACKERS)}")
    print(f"   登録済みトラッカー数: {len(trackers)}")
    print(f"   登録済みファイル: {list(trackers.keys())}")

def get_version_count():
    """現在登録されているバージョン数を取得"""
    return VersionTracker.get_tracker_count()
