"""SQLite persistence for TuntunClaw (囤囤钳) booth inventory, scan logs, recycle logs, and replenishment checklist."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


DEFAULT_ITEMS = [
    ("cuicuisha",  "雀巢脆脆鲨", "Nestle Wafer 15g",    "零食", 2, 1, 4, "条", "#8a221c"),
    ("toothpaste", "舒客+小牙膏", "Saky Travel Paste 20g", "洗护", 3, 1, 5, "支", "#2c89a6"),
]

SKU_ALIASES = {
    "coffee": "cuicuisha",
    "wafer": "cuicuisha",
    "oreo": "toothpaste",
    "paste": "toothpaste",
}


class TuntunDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._lock, closing(self._conn()) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS items (
                    key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_en TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '展台物资',
                    quantity INTEGER NOT NULL DEFAULT 0,
                    delivered INTEGER NOT NULL DEFAULT 0,
                    threshold INTEGER NOT NULL DEFAULT 1,
                    reorder_qty INTEGER NOT NULL DEFAULT 5,
                    unit TEXT NOT NULL DEFAULT '个',
                    color_hex TEXT NOT NULL DEFAULT '#087f78',
                    last_seen_pos TEXT DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS replenishment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_key TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    suggested_qty INTEGER NOT NULL DEFAULT 5,
                    unit TEXT NOT NULL DEFAULT '个',
                    reason TEXT NOT NULL DEFAULT '低于安全库存阈值',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            cnt = db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            has_new = db.execute("SELECT 1 FROM items WHERE key = 'cuicuisha'").fetchone()
            if cnt > 0 and not has_new:
                db.execute("DELETE FROM items WHERE key IN ('coffee', 'oreo', 'apple', 'orange')")
                db.execute("DELETE FROM replenishment")
                cnt = db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            if cnt == 0:
                now = _now()
                db.executemany(
                    "INSERT INTO items(key, name, name_en, category, quantity, delivered, threshold, reorder_qty, unit, color_hex, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
                    [(*row, now) for row in DEFAULT_ITEMS],
                )
                db.execute(
                    "INSERT INTO events(event_type, title, detail, created_at) VALUES (?, ?, ?, ?)",
                    ("system", "TuntunClaw 囤囤钳系统初始化", "已载入 1:1 展台物料 (脆脆鲨×2, 舒客小牙膏×3)", now),
                )
            db.commit()
        self.sync_replenishment_rules()

    def get_item(self, item_key: str) -> dict[str, Any] | None:
        item_key = SKU_ALIASES.get(item_key, item_key)
        with self._lock, closing(self._conn()) as db:
            row = db.execute("SELECT * FROM items WHERE key = ?", (item_key,)).fetchone()
            if not row:
                return None
            return {**dict(row), "low_stock": row["quantity"] <= row["threshold"]}

    def sync_replenishment_rules(self) -> list[dict[str, Any]]:
        now = _now()
        newly_added = []
        with self._lock, closing(self._conn()) as db:
            rows = db.execute("SELECT * FROM items ORDER BY rowid").fetchall()
            for r in rows:
                if r["quantity"] <= r["threshold"]:
                    existing = db.execute(
                        "SELECT id FROM replenishment WHERE item_key = ? AND status = 'pending'",
                        (r["key"],),
                    ).fetchone()
                    if not existing:
                        reason = f"当前余量 {r['quantity']}{r['unit']} ≤ 阈值 {r['threshold']}{r['unit']}"
                        db.execute(
                            "INSERT INTO replenishment(item_key, item_name, suggested_qty, unit, reason, status, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                            (r["key"], r["name"], r["reorder_qty"], r["unit"], reason, now, now),
                        )
                        newly_added.append({"item_key": r["key"], "item_name": r["name"], "reason": reason})
                else:
                    db.execute(
                        "UPDATE replenishment SET status = 'done', updated_at = ? WHERE item_key = ? AND status = 'pending'",
                        (now, r["key"]),
                    )
            db.commit()
        return newly_added

    def add_event(self, event_type: str, title: str, detail: str = "") -> None:
        with self._lock, closing(self._conn()) as db:
            db.execute(
                "INSERT INTO events(event_type, title, detail, created_at) VALUES (?, ?, ?, ?)",
                (event_type, title, detail, _now()),
            )
            db.commit()

    def update_from_scan(self, counts: dict[str, int], positions: dict[str, list[list[float]]]) -> dict[str, Any]:
        now = _now()
        with self._lock, closing(self._conn()) as db:
            for key, qty in counts.items():
                pos_str = json.dumps(positions.get(key, []))
                db.execute(
                    "UPDATE items SET quantity = ?, last_seen_pos = ?, updated_at = ? WHERE key = ?",
                    (int(qty), pos_str, now, key),
                )
            db.commit()
        summary = ", ".join(f"{k}:{v}" for k, v in counts.items())
        self.add_event("scan", "完成桌面场景视觉扫描", f"腕部RGB+顶视识别结果: {summary}")
        alerts = self.sync_replenishment_rules()
        if alerts:
            self.add_event(
                "alert",
                f"触发 {len(alerts)} 项补货预警",
                "；".join(f"{a['item_name']}({a['reason']})" for a in alerts),
            )
        return self.snapshot()

    def record_delivery(self, item_key: str, source: str = "sim_pick") -> dict[str, Any]:
        item_key = SKU_ALIASES.get(item_key, item_key)
        now = _now()
        with self._lock, closing(self._conn()) as db:
            row = db.execute("SELECT * FROM items WHERE key = ?", (item_key,)).fetchone()
            if not row:
                raise KeyError(item_key)
            new_qty = max(0, row["quantity"] - 1)
            new_deliv = row["delivered"] + 1
            db.execute(
                "UPDATE items SET quantity = ?, delivered = ?, updated_at = ? WHERE key = ?",
                (new_qty, new_deliv, now, item_key),
            )
            db.commit()
        self.add_event(
            "pick",
            f"已抓取并递送 1 {row['unit']}{row['name']}",
            f"来源: {source} | 剩余 {new_qty}{row['unit']} (阈值 {row['threshold']}{row['unit']})",
        )
        alerts = self.sync_replenishment_rules()
        if alerts:
            self.add_event(
                "alert",
                f"{row['name']} 触发自动补货清单",
                "；".join(a["reason"] for a in alerts),
            )
        return self.snapshot()

    def record_recycle(self, item_key: str) -> dict[str, Any]:
        item_key = SKU_ALIASES.get(item_key, item_key)
        now = _now()
        with self._lock, closing(self._conn()) as db:
            row = db.execute("SELECT * FROM items WHERE key = ?", (item_key,)).fetchone()
            if not row:
                raise KeyError(item_key)
            new_qty = row["quantity"] + 1
            new_deliv = max(0, row["delivered"] - 1)
            db.execute(
                "UPDATE items SET quantity = ?, delivered = ?, updated_at = ? WHERE key = ?",
                (new_qty, new_deliv, now, item_key),
            )
            db.commit()
        self.add_event(
            "recycle",
            f"已回收归位 1 {row['unit']}{row['name']}",
            f"取料区余量恢复为 {new_qty}{row['unit']}",
        )
        self.sync_replenishment_rules()
        return self.snapshot()

    def set_replenishment_status(self, rep_id: int, status: str) -> dict[str, Any]:
        now = _now()
        with self._lock, closing(self._conn()) as db:
            row = db.execute("SELECT * FROM replenishment WHERE id = ?", (rep_id,)).fetchone()
            if row:
                db.execute(
                    "UPDATE replenishment SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, rep_id),
                )
                db.commit()
                self.add_event("replenish", f"补货项更新: {row['item_name']}", f"状态 -> {status}")
        return self.snapshot()

    def add_custom_replenishment(self, item_name: str, qty: int = 2, unit: str = "件", reason: str = "展台手动登记") -> dict[str, Any]:
        now = _now()
        with self._lock, closing(self._conn()) as db:
            db.execute(
                "INSERT INTO replenishment(item_key, item_name, suggested_qty, unit, reason, status, created_at, updated_at) "
                "VALUES ('custom', ?, ?, ?, ?, 'pending', ?, ?)",
                (item_name.strip(), max(1, int(qty)), unit.strip() or "件", reason, now, now),
            )
            db.commit()
        self.add_event("replenish", f"新增补货需求: {item_name}", f"数量 {qty}{unit} | {reason}")
        return self.snapshot()

    def update_threshold(self, item_key: str, threshold: int) -> dict[str, Any]:
        item_key = SKU_ALIASES.get(item_key, item_key)
        now = _now()
        with self._lock, closing(self._conn()) as db:
            db.execute(
                "UPDATE items SET threshold = ?, updated_at = ? WHERE key = ?",
                (max(0, int(threshold)), now, item_key),
            )
            db.commit()
        self.sync_replenishment_rules()
        return self.snapshot()

    def reset_scene_stock(self) -> dict[str, Any]:
        now = _now()
        with self._lock, closing(self._conn()) as db:
            for key, name, name_en, cat, qty, th, rq, unit, col in DEFAULT_ITEMS:
                db.execute(
                    "UPDATE items SET quantity = ?, delivered = 0, threshold = ?, updated_at = ? WHERE key = ?",
                    (qty, th, now, key),
                )
            db.execute("DELETE FROM replenishment")
            db.commit()
        self.sync_replenishment_rules()
        self.add_event("restock", "展台取料区已一键补货复位", "5 件 1:1 实景物料（脆脆鲨×2、舒客小牙膏×3）已重新就位")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock, closing(self._conn()) as db:
            items = [
                {**dict(r), "low_stock": r["quantity"] <= r["threshold"]}
                for r in db.execute("SELECT * FROM items ORDER BY rowid").fetchall()
            ]
            replenishment = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM replenishment ORDER BY CASE status WHEN 'pending' THEN 0 WHEN 'ordered' THEN 1 ELSE 2 END, id DESC LIMIT 30"
                ).fetchall()
            ]
            events = [
                dict(r)
                for r in db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 25").fetchall()
            ]
        return {
            "items": items,
            "replenishment": replenishment,
            "events": events,
            "summary": {
                "total_stock": sum(i["quantity"] for i in items),
                "total_delivered": sum(i["delivered"] for i in items),
                "low_stock_types": sum(1 for i in items if i["low_stock"]),
                "pending_replenishment": sum(1 for r in replenishment if r["status"] == "pending"),
            },
            "server_time": _now(),
        }
