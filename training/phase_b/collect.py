#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# -----------------------------
# Paths / Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "phase_b"

CONFIG = {
    "default_label": "human",
    "out_root": DEFAULT_OUT_ROOT,  # --out-dir 로 덮어씀
}

# ----------------------------------------------------------------
# HTML & JavaScript
# ----------------------------------------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>T-curity Phase B Collector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; user-select: none; -webkit-tap-highlight-color: transparent; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; color: #fff; min-height: 100vh;
            display: flex; flex-direction: column; align-items: center; padding: 15px;
            touch-action: none;
        }
        h1 { font-size: 1.2rem; margin-bottom: 8px; color: #8be9fd; }
        .stats { font-size: 0.85rem; margin-bottom: 8px; color: #aaa; }
        .instruction {
            background: rgba(255,255,255,0.1); padding: 15px; border-radius: 12px;
            margin-bottom: 15px; text-align: center; width: 100%; max-width: 400px;
        }
        .target-class { font-size: 1.4rem; color: #50fa7b; font-weight: bold; margin: 5px 0; }

        .grid-panel { background: rgba(255,255,255,0.05); border-radius: 15px; padding: 15px; margin-bottom: 20px; }
        .image-grid { display: grid; grid-template-columns: repeat(3, 90px); gap: 10px; }
        .grid-cell {
            width: 90px; height: 90px; background: rgba(255,255,255,0.1);
            border: 2px solid rgba(255,255,255,0.2); border-radius: 12px;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            position: relative; cursor: grab;
        }
        .grid-cell.placed { opacity: 0.2; cursor: not-allowed; border-style: dashed; }
        .emoji { font-size: 1.8rem; }
        .order-hint { font-size: 0.75rem; color: #8be9fd; margin-top: 2px; font-weight: bold; }

        .slot-panel {
            background: rgba(80, 250, 123, 0.05); border: 1px solid rgba(80, 250, 123, 0.2);
            border-radius: 15px; padding: 15px; width: 100%; max-width: 400px;
        }
        .slot-grid { display: flex; justify-content: space-between; gap: 8px; }
        .slot {
            flex: 1; height: 80px; background: rgba(80, 250, 123, 0.1);
            border: 2px dashed rgba(80, 250, 123, 0.3); border-radius: 10px;
            display: flex; align-items: center; justify-content: center; font-size: 1.8rem;
            position: relative;
        }
        .slot-num { position: absolute; bottom: 2px; right: 5px; font-size: 0.65rem; color: #50fa7b; font-weight: bold; }
        .slot.filled { border-style: solid; background: rgba(80, 250, 123, 0.2); }

        .slot .remove-btn{
        position:absolute;
        top:6px; left:6px;
        width:22px; height:22px;
        border-radius:999px;
        display:flex; align-items:center; justify-content:center;
        font-size:14px; font-weight:700;
        background: rgba(0,0,0,0.55);
        color:#fff;
        opacity:0;
        transform: scale(0.95);
        transition: opacity .12s ease, transform .12s ease;
        cursor:pointer;
        }

        .slot.filled:hover .remove-btn { opacity:1; transform: scale(1); }
        /* 모바일(hover 없음) 대비: filled면 항상 보이게 하고 싶으면 아래 주석 해제 */
        /* .slot.filled .remove-btn { opacity:1; transform: scale(1); } */

        .label-selector { margin: 15px 0; display: flex; gap: 10px; }
        .label-btn { padding: 10px 20px; border-radius: 8px; border: 1px solid #4a4a6a; background: #2a2a4a; color: #ccc; cursor: pointer; }
        .label-btn.active { background: #50fa7b; color: #1a1a2e; border-color: #50fa7b; font-weight: bold; }

        .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 100; align-items: center; justify-content: center; }
        .modal.show { display: flex; }
    </style>
</head>
<body>
    <h1>🎫 Phase B Collector</h1>

    <div class="stats">
        Server: <span id="serverCount">0</span> | Session: <span id="sessionCount">0</span> | Points: <span id="pointCount">0</span>
    </div>

    <div class="instruction">
        <p>다음 이미지를 찾으세요</p>
        <div class="target-class" id="targetClass">로딩 중...</div>
        <p><strong>1~4번 숫자</strong>가 적힌 이미지를 <strong>순서대로</strong> 넣으세요</p>
    </div>

    <div class="grid-panel">
        <div class="image-grid" id="imageGrid"></div>
    </div>

    <div class="slot-panel">
        <div class="slot-grid" id="slotGrid"></div>
    </div>

    <div class="label-selector">
        <button class="label-btn active" id="btnHuman" onclick="setLabel('human')">👤 Human</button>
        <button class="label-btn" id="btnBot" onclick="setLabel('bot')">🤖 Bot</button>
    </div>

    <div class="modal" id="resultModal">
        <div style="text-align:center">
            <h2 id="modalTitle" style="font-size:2rem; margin-bottom:20px"></h2>
            <p id="modalDetail" style="margin-bottom:20px; color:#aaa;"></p>
            <button onclick="resetGame()" style="padding:12px 30px; border-radius:8px; background:#8be9fd; border:none; cursor:pointer; font-weight:bold;">다음 문제</button>
        </div>
    </div>

    <script>
        const CLASSES = {
            cat: '🐱', dog: '🐕', bird: '🐦', car: '🚗', plane: '✈️', ship: '🚢',

            // 음식
            pizza: '🍕', burger: '🍔', sushi: '🍣', cake: '🍰', coffee: '☕️',

            // 동물
            rabbit: '🐰', panda: '🐼', tiger: '🐯', frog: '🐸', penguin: '🐧',

            // 자연/날씨
            sun: '☀️', moon: '🌙', star: '⭐️', cloud: '☁️', snow: '❄️',

            // 스포츠/취미
            soccer: '⚽️', basketball: '🏀', guitar: '🎸', game: '🎮'
        };

        let state = { target: '', grid: [], slots: [null, null, null, null], points: [], startTime: 0, label: 'human', session: 0 };
        let draggingItem = null;

        function initGame() {
            const keys = Object.keys(CLASSES);
            state.target = keys[Math.floor(Math.random() * keys.length)];
            state.grid = [];

            // ✅ target 제외 후보군은 여기서 한번만 만들기
            const others = keys.filter(k => k !== state.target);

            let nums = [1,2,3,4,5,6,7,8,9].sort(() => Math.random() - 0.5);

            for (let i = 0; i < 9; i++) {
                const num = nums[i];
                const isTargetRequired = num <= 4;

                state.grid.push({
                type: isTargetRequired
                    ? state.target
                    : others[Math.floor(Math.random() * others.length)],
                displayNum: num,
                id: 'img' + i,
                placed: false
                });
            }

            state.grid.sort(() => Math.random() - 0.5);

            state.slots = [null, null, null, null];
            state.points = [];
            state.startTime = 0;

            render();
            document.getElementById('resultModal').classList.remove('show');
        }


        function render() {
            document.getElementById('targetClass').textContent = CLASSES[state.target] + ' ' + state.target.toUpperCase();
            const gridEl = document.getElementById('imageGrid');
            gridEl.innerHTML = '';
            state.grid.forEach((item) => {
                const cell = document.createElement('div');
                cell.className = 'grid-cell' + (item.placed ? ' placed' : '');
                cell.innerHTML = `<span class="emoji">${CLASSES[item.type]}</span><span class="order-hint">${item.displayNum}</span>`;

                if(!item.placed) {
                    cell.draggable = true;
                    const startAction = () => {
                        if(state.startTime === 0) {
                            state.startTime = performance.now();
                            trackMouse();
                        }
                        draggingItem = item;
                    };
                    cell.addEventListener('dragstart', startAction);
                    cell.addEventListener('touchstart', (e) => {
                        e.preventDefault();
                        startAction();
                    }, {passive: false});
                }
                gridEl.appendChild(cell);
            });

            const slotEl = document.getElementById('slotGrid');
            slotEl.innerHTML = '';
            state.slots.forEach((item, idx) => {
            const slot = document.createElement('div');
            slot.className = 'slot' + (item ? ' filled' : '');

            // ✅ filled면 X 버튼 표시 + 클릭 삭제
            if(item) {
                slot.innerHTML = `
                ${CLASSES[item.type]}
                <span class="slot-num">${idx + 1}</span>
                <span class="remove-btn" title="삭제">✕</span>
                `;

                // X 눌렀을 때 삭제
                slot.querySelector('.remove-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                removeFromSlot(idx);
                });

                // 슬롯 자체 클릭해도 삭제하고 싶으면 (원하면)
                slot.addEventListener('click', () => removeFromSlot(idx));

            } else {
                slot.innerHTML = `${''}<span class="slot-num">${idx + 1}</span>`;
            }

            slot.addEventListener('dragover', e => e.preventDefault());
            slot.addEventListener('drop', () => handleDrop(idx));

            slotEl.appendChild(slot);
            });

        }

        function trackMouse() {
            const tracker = (e) => {
                if(state.startTime === 0) return;
                const t = e.touches ? e.touches[0] : e;
                state.points.push({
                    x: (t.clientX / window.innerWidth).toFixed(4),
                    y: (t.clientY / window.innerHeight).toFixed(4),
                    t: Math.round(performance.now() - state.startTime)
                });
                document.getElementById('pointCount').textContent = state.points.length;
            };
            window.onmousemove = tracker;
            window.ontouchmove = tracker;
        }

        async function handleDrop(slotIdx) {
            if(!draggingItem || state.slots[slotIdx]) return;
            state.slots[slotIdx] = draggingItem;
            state.grid.find(g => g.id === draggingItem.id).placed = true;
            draggingItem = null;
            render();
            if(state.slots.every(s => s !== null)) { await submit(); }
        }

        window.ontouchend = (e) => {
            if(!draggingItem) return;
            const touch = e.changedTouches[0];
            const slot = document.elementsFromPoint(touch.clientX, touch.clientY).find(el => el.classList.contains('slot'));
            if(slot) { handleDrop(Array.from(slot.parentNode.children).indexOf(slot)); }
            draggingItem = null;
        };

        async function submit() {
            let correct = 0;
            state.slots.forEach((item, i) => {
                if (item && item.type === state.target && item.displayNum === (i + 1)) correct++;
            });

            const isPerfect = (correct === 4);
            const payload = {
                label: state.label,
                timestamp: new Date().toISOString(),
                target: state.target,
                correct_count: correct,
                is_perfect: isPerfect,
                points: state.points,
                metadata: { ua: navigator.userAgent, res: {w: window.innerWidth, h: window.innerHeight} }
            };

            await fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            state.session++;
            document.getElementById('sessionCount').textContent = state.session;
            fetchStats();

            document.getElementById('modalTitle').textContent = isPerfect ? '✅ 성공!' : '❌ 실패';
            document.getElementById('modalDetail').textContent = `정확도: ${correct}/4 (숫자 순서대로 넣으셨나요?)`;
            document.getElementById('resultModal').classList.add('show');
        }

        function setLabel(l) {
            state.label = l;
            document.getElementById('btnHuman').classList.toggle('active', l === 'human');
            document.getElementById('btnBot').classList.toggle('active', l === 'bot');
        }

        async function fetchStats() {
            try {
                const res = await fetch('/stats');
                const data = await res.json();
                document.getElementById('serverCount').textContent = data.count;
            } catch(e) { console.error(e); }
        }

        function resetGame() { initGame(); }
        window.onload = () => { initGame(); fetchStats(); };

        function removeFromSlot(slotIdx) {
            const item = state.slots[slotIdx];
            if(!item) return;

            // 슬롯 비우기
            state.slots[slotIdx] = null;

            // 그리드에서 다시 사용 가능하도록 placed 해제
            const g = state.grid.find(x => x.id === item.id);
            if(g) g.placed = false;

            render();
        }
    </script>
</body>
</html>
"""

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/save", methods=["POST"])
def save():
    try:
        data = request.get_json(force=True) or {}
        label = (data.get("label") or CONFIG["default_label"]).strip().lower()

        # 라벨별 폴더로 저장: OUT_ROOT / human|bot
        out_dir = Path(CONFIG["out_root"]) / label
        out_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{timestamp}_{label}.json"

        filepath = out_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({"ok": True, "file": filename, "dir": str(out_dir)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/stats")
def stats():
    out_root = Path(CONFIG["out_root"])
    total = 0
    by_label = {}

    if out_root.exists():
        # 라벨 폴더(human/bot 등) 전체 합산
        for p in out_root.rglob("*.json"):
            total += 1
            lbl = p.parent.name
            by_label[lbl] = by_label.get(lbl, 0) + 1

    # 프론트는 기존처럼 count만 사용해도 됨
    return jsonify({"count": total, "by_label": by_label, "out_root": str(out_root)})


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-curity Phase B Collector")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_ROOT),
        help="데이터 저장 루트 경로 (예: ./data/phase_b). 실제 저장은 out-dir/<label>/ 아래로 됨.",
    )
    parser.add_argument("--port", type=int, default=5002, help="포트 번호")
    args = parser.parse_args()

    CONFIG["out_root"] = Path(args.out_dir).expanduser().resolve()

    print("=" * 50)
    print(" 🧩 T-curity Phase B Collector Server Started")
    print(f" - OUT_ROOT: {CONFIG['out_root']}")
    print(f" - EX) human -> {CONFIG['out_root'] / 'human'}")
    print(f" - URL: http://localhost:{args.port}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=args.port, debug=False)