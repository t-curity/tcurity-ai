# generate_hard_bots.py
import json
import random
import os

def generate_bezier_bot():
    """베지어 곡선 봇"""
    points = []
    base_time = 0
    for drag_idx in range(4):
        start_x = 0.3 + (drag_idx % 3) * 0.2 + random.uniform(-0.05, 0.05)
        start_y = 0.35 + random.uniform(-0.03, 0.03)
        end_y = 0.75 + random.uniform(-0.03, 0.03)
        num_points = random.randint(18, 25)
        for i in range(num_points):
            t = i / (num_points - 1)
            ease_t = t * t * (3 - 2 * t)
            points.append({
                "x": start_x + random.uniform(-0.003, 0.003),
                "y": start_y + (end_y - start_y) * ease_t,
                "t": base_time + i * random.randint(16, 20)
            })
        base_time += num_points * 18 + random.randint(600, 900)
    return {"points": points}

def generate_recorded_replay_bot():
    """녹화 리플레이 봇"""
    points = []
    base_time = 0
    human_patterns = [
        [(0.35, 0), (0.36, 18), (0.38, 35), (0.41, 52), (0.45, 68), (0.50, 85), (0.54, 105), (0.58, 128), (0.63, 152), (0.67, 180), (0.70, 210), (0.72, 245), (0.74, 285), (0.75, 320)],
        [(0.35, 0), (0.37, 20), (0.40, 42), (0.44, 65), (0.49, 90), (0.55, 118), (0.60, 150), (0.65, 185), (0.69, 225), (0.72, 270), (0.75, 320)],
        [(0.35, 0), (0.38, 25), (0.42, 55), (0.47, 88), (0.53, 125), (0.59, 165), (0.64, 210), (0.69, 260), (0.73, 310), (0.75, 350)],
    ]
    for drag_idx in range(4):
        start_x = 0.3 + (drag_idx % 3) * 0.2 + random.uniform(-0.05, 0.05)
        pattern = random.choice(human_patterns)
        for y, dt in pattern:
            x_jitter = random.uniform(-0.008, 0.008)
            y_jitter = random.uniform(-0.005, 0.005)
            points.append({"x": start_x + x_jitter, "y": y + y_jitter, "t": base_time + dt + random.randint(-5, 5)})
        base_time += pattern[-1][1] + random.randint(600, 1000)
    return {"points": points}

def generate_acceleration_bot():
    """가속도 패턴 봇"""
    points = []
    base_time = 0
    for drag_idx in range(4):
        start_x = 0.3 + (drag_idx % 3) * 0.2 + random.uniform(-0.05, 0.05)
        start_y = 0.35 + random.uniform(-0.03, 0.03)
        end_y = 0.75 + random.uniform(-0.03, 0.03)
        steps = random.randint(20, 28)
        t = base_time
        for i in range(steps):
            progress = i / (steps - 1)
            ease = progress * progress * (3 - 2 * progress)
            y = start_y + (end_y - start_y) * ease
            if progress < 0.3:
                dt = random.randint(20, 30)
            elif progress > 0.7:
                dt = random.randint(18, 28)
            else:
                dt = random.randint(10, 18)
            x_jitter = random.gauss(0, 0.008)
            points.append({"x": start_x + x_jitter, "y": y, "t": t})
            t += dt
        base_time = t + random.randint(600, 1000)
    return {"points": points}

def generate_variable_speed_bot():
    """가변 속도 봇"""
    points = []
    base_time = 0
    for drag_idx in range(4):
        start_x = 0.3 + (drag_idx % 3) * 0.2 + random.uniform(-0.05, 0.05)
        start_y = 0.35 + random.uniform(-0.03, 0.03)
        end_y = 0.75 + random.uniform(-0.03, 0.03)
        y = start_y
        t = base_time
        while y < end_y:
            if random.random() < 0.3:
                speed = random.uniform(0.005, 0.015)
                dt = random.randint(25, 40)
            else:
                speed = random.uniform(0.02, 0.04)
                dt = random.randint(10, 18)
            x_jitter = random.uniform(-0.012, 0.012)
            points.append({"x": start_x + x_jitter, "y": y, "t": t})
            y += speed
            t += dt
        base_time = t + random.randint(600, 1000)
    return {"points": points}

def generate_selenium_basic_bot():
    """Selenium 기본 봇"""
    points = []
    base_time = 0
    for drag_idx in range(4):
        start_x = 0.3 + (drag_idx % 3) * 0.2 + random.uniform(-0.05, 0.05)
        start_y = 0.35 + random.uniform(-0.03, 0.03)
        end_y = 0.75 + random.uniform(-0.03, 0.03)
        num_steps = random.randint(8, 15)
        for i in range(num_steps):
            points.append({
                "x": start_x,
                "y": start_y + (end_y - start_y) * i / (num_steps - 1),
                "t": base_time + i * random.randint(50, 100)
            })
        base_time += num_steps * 75 + random.randint(400, 800)
    return {"points": points}

# 봇 데이터 저장
bot_dir = "/home/ubuntu/tcurity-ai/data/phase_b/bot"
os.makedirs(bot_dir, exist_ok=True)

generators = [
    (generate_bezier_bot, "bezier"),
    (generate_recorded_replay_bot, "replay"),
    (generate_acceleration_bot, "acceleration"),
    (generate_variable_speed_bot, "variable"),
    (generate_selenium_basic_bot, "selenium"),
]

total = 0
for gen_func, name in generators:
    for i in range(100):  # 각 유형 100개
        data = gen_func()
        with open(f"{bot_dir}/{name}_bot_{i:03d}.json", "w") as f:
            json.dump(data, f)
        total += 1

print(f"봇 데이터 {total}개 생성 완료")
print(f"총 봇 데이터: {len(os.listdir(bot_dir))}개")