# test_phase_b_bot_stress.py
import requests
import random
from collections import defaultdict

AI_SERVER = "http://localhost:9000"

class PhaseBBotTester:
    def generate_bot_drag(self, bot_type="uniform"):
        points = []
        base_time = 0
        
        for drag_idx in range(4):
            start_x = 0.3 + (drag_idx % 3) * 0.2 + random.uniform(-0.05, 0.05)
            start_y = 0.35 + random.uniform(-0.03, 0.03)
            end_y = 0.75 + random.uniform(-0.03, 0.03)
            
            if bot_type == "uniform":
                interval = random.randint(14, 18)
                for i in range(20):
                    points.append({
                        "x": start_x,
                        "y": start_y + (end_y - start_y) * i / 19,
                        "t": base_time + i * interval
                    })
                base_time += 20 * interval + random.randint(600, 1000)
                
            elif bot_type == "linear":
                num_points = random.randint(12, 18)
                interval = random.randint(18, 25)
                for i in range(num_points):
                    points.append({
                        "x": start_x,
                        "y": start_y + (end_y - start_y) * i / (num_points - 1),
                        "t": base_time + i * interval
                    })
                base_time += num_points * interval + random.randint(500, 900)
                
            elif bot_type == "perfect_timing":
                for i in range(25):
                    points.append({
                        "x": start_x + random.uniform(-0.002, 0.002),
                        "y": start_y + (end_y - start_y) * i / 24,
                        "t": base_time + i * 16
                    })
                base_time += 400 + random.randint(500, 800)
                
            elif bot_type == "bezier":
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
                
            elif bot_type == "jitter":
                for i in range(20):
                    points.append({
                        "x": start_x + random.uniform(-0.015, 0.015),
                        "y": start_y + (end_y - start_y) * i / 19 + random.uniform(-0.008, 0.008),
                        "t": base_time + i * 16 + random.randint(-3, 3)
                    })
                base_time += 320 + random.randint(600, 1000)
                
            elif bot_type == "smart_human":
                y = start_y
                t = base_time
                while y < end_y:
                    speed = random.uniform(0.008, 0.035)
                    dt = random.randint(12, 28)
                    x_jitter = random.uniform(-0.015, 0.015)
                    points.append({"x": start_x + x_jitter, "y": y, "t": t})
                    y += speed
                    t += dt
                base_time = t + random.randint(500, 1200)

            elif bot_type == "recorded_replay":
                human_patterns = [
                    [(0.35, 0), (0.36, 18), (0.38, 35), (0.41, 52), (0.45, 68), (0.50, 85), (0.54, 105), (0.58, 128), (0.63, 152), (0.67, 180), (0.70, 210), (0.72, 245), (0.74, 285), (0.75, 320)],
                    [(0.35, 0), (0.37, 20), (0.40, 42), (0.44, 65), (0.49, 90), (0.55, 118), (0.60, 150), (0.65, 185), (0.69, 225), (0.72, 270), (0.75, 320)],
                    [(0.35, 0), (0.38, 25), (0.42, 55), (0.47, 88), (0.53, 125), (0.59, 165), (0.64, 210), (0.69, 260), (0.73, 310), (0.75, 350)],
                ]
                pattern = random.choice(human_patterns)
                for y, dt in pattern:
                    x_jitter = random.uniform(-0.008, 0.008)
                    y_jitter = random.uniform(-0.005, 0.005)
                    points.append({"x": start_x + x_jitter, "y": y + y_jitter, "t": base_time + dt + random.randint(-5, 5)})
                base_time += pattern[-1][1] + random.randint(600, 1000)

            elif bot_type == "ml_evasion":
                y = start_y
                t = base_time
                while y < end_y:
                    base_speed = 0.02
                    speed = base_speed + random.gauss(0, 0.008)
                    speed = max(0.005, min(0.04, speed))
                    dt = random.randint(14, 24)
                    x_jitter = random.gauss(0, 0.01)
                    points.append({"x": start_x + x_jitter, "y": y, "t": t})
                    y += speed
                    t += dt
                base_time = t + random.randint(700, 1100)

            elif bot_type == "pause_human":
                y = start_y
                t = base_time
                pause_point = random.uniform(0.45, 0.60)
                pause_added = False
                while y < end_y:
                    speed = random.uniform(0.01, 0.03)
                    dt = random.randint(14, 22)
                    if not pause_added and y > pause_point:
                        dt = random.randint(150, 350)
                        pause_added = True
                    x_jitter = random.uniform(-0.01, 0.01)
                    points.append({"x": start_x + x_jitter, "y": y, "t": t})
                    y += speed
                    t += dt
                base_time = t + random.randint(600, 1000)

            elif bot_type == "acceleration":
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

            elif bot_type == "variable_speed":
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

            elif bot_type == "micro_jitter":
                num_points = random.randint(20, 26)
                for i in range(num_points):
                    progress = i / (num_points - 1)
                    y = start_y + (end_y - start_y) * progress
                    dt = 16 + random.randint(-3, 3)
                    x_jitter = random.gauss(0, 0.003)
                    y_jitter = random.gauss(0, 0.002)
                    points.append({"x": start_x + x_jitter, "y": y + y_jitter, "t": base_time})
                    base_time += dt
                base_time += random.randint(600, 1000)

            elif bot_type == "selenium_basic":
                num_steps = random.randint(8, 15)
                for i in range(num_steps):
                    points.append({
                        "x": start_x,
                        "y": start_y + (end_y - start_y) * i / (num_steps - 1),
                        "t": base_time + i * random.randint(50, 100)
                    })
                base_time += num_steps * 75 + random.randint(400, 800)

            elif bot_type == "pyautogui":
                duration = random.randint(200, 400)
                num_points = random.randint(10, 20)
                for i in range(num_points):
                    points.append({
                        "x": start_x + random.uniform(-0.002, 0.002),
                        "y": start_y + (end_y - start_y) * i / (num_points - 1),
                        "t": base_time + int(duration * i / (num_points - 1))
                    })
                base_time += duration + random.randint(500, 900)

            # 추가 봇 유형들
            elif bot_type == "super_human":
                # 사람보다 더 사람같은 봇
                y = start_y
                t = base_time
                while y < end_y:
                    speed = random.gauss(0.018, 0.012)
                    speed = max(0.003, min(0.045, speed))
                    dt = int(random.gauss(18, 5))
                    dt = max(8, min(35, dt))
                    x_jitter = random.gauss(0, 0.012)
                    points.append({"x": start_x + x_jitter, "y": y, "t": t})
                    y += speed
                    t += dt
                base_time = t + random.randint(500, 1200)

            elif bot_type == "neural_mimic":
                # 신경망 학습 흉내 봇
                y = start_y
                t = base_time
                momentum = 0
                while y < end_y:
                    target_speed = random.gauss(0.02, 0.008)
                    momentum = momentum * 0.7 + target_speed * 0.3
                    speed = max(0.005, min(0.04, momentum))
                    dt = int(16 + random.gauss(0, 4))
                    dt = max(10, min(28, dt))
                    x_drift = random.gauss(0, 0.008)
                    points.append({"x": start_x + x_drift, "y": y, "t": t})
                    y += speed
                    t += dt
                base_time = t + random.randint(600, 1100)

            elif bot_type == "hesitation":
                # 망설임 봇 (중간중간 느려짐)
                y = start_y
                t = base_time
                while y < end_y:
                    if random.random() < 0.15:
                        speed = random.uniform(0.002, 0.008)
                        dt = random.randint(40, 80)
                    else:
                        speed = random.uniform(0.015, 0.03)
                        dt = random.randint(14, 22)
                    x_jitter = random.uniform(-0.01, 0.01)
                    points.append({"x": start_x + x_jitter, "y": y, "t": t})
                    y += speed
                    t += dt
                base_time = t + random.randint(600, 1000)

            elif bot_type == "overshoot":
                # 오버슈트 봇 (목표 지나쳤다가 돌아옴)
                y = start_y
                t = base_time
                overshot = False
                while True:
                    if not overshot and y >= end_y:
                        overshot = True
                        overshoot_y = y + random.uniform(0.02, 0.05)
                    if overshot and y <= end_y:
                        break
                    if overshot:
                        speed = -random.uniform(0.01, 0.02)
                    else:
                        speed = random.uniform(0.015, 0.03)
                    dt = random.randint(14, 22)
                    x_jitter = random.uniform(-0.008, 0.008)
                    points.append({"x": start_x + x_jitter, "y": y, "t": t})
                    y += speed
                    t += dt
                    if t - base_time > 2000:
                        break
                base_time = t + random.randint(600, 1000)

            elif bot_type == "curved_path":
                # 곡선 경로 봇
                num_points = random.randint(18, 28)
                curve_amount = random.uniform(0.03, 0.08)
                for i in range(num_points):
                    progress = i / (num_points - 1)
                    y = start_y + (end_y - start_y) * progress
                    x_curve = curve_amount * math.sin(progress * math.pi)
                    dt = random.randint(14, 22)
                    points.append({"x": start_x + x_curve + random.gauss(0, 0.005), "y": y, "t": base_time})
                    base_time += dt
                base_time += random.randint(600, 1000)

            elif bot_type == "stutter":
                # 끊김 봇 (가다 멈추다 반복)
                y = start_y
                t = base_time
                moving = True
                while y < end_y:
                    if moving:
                        speed = random.uniform(0.02, 0.035)
                        dt = random.randint(12, 18)
                        if random.random() < 0.2:
                            moving = False
                    else:
                        speed = 0
                        dt = random.randint(30, 60)
                        if random.random() < 0.5:
                            moving = True
                    x_jitter = random.uniform(-0.008, 0.008)
                    points.append({"x": start_x + x_jitter, "y": y, "t": t})
                    y += speed
                    t += dt
                base_time = t + random.randint(600, 1000)
        
        return points

    def test_bot(self, points):
        try:
            resp = requests.post(
                f"{AI_SERVER}/test/phase_b",
                json={"points": points},
                timeout=10
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e), "pass": True, "score": 0}


import math

def main():
    tester = PhaseBBotTester()
    
    bot_types = [
        ("uniform", "균일 속도 봇"),
        ("linear", "선형 보간 봇"),
        ("perfect_timing", "완벽 타이밍 봇"),
        ("bezier", "베지어 곡선 봇"),
        ("jitter", "지터 추가 봇"),
        ("smart_human", "스마트 사람흉내 봇"),
        ("recorded_replay", "녹화 리플레이 봇"),
        ("ml_evasion", "ML 회피 봇"),
        ("pause_human", "멈춤 패턴 봇"),
        ("acceleration", "가속도 패턴 봇"),
        ("variable_speed", "가변 속도 봇"),
        ("micro_jitter", "미세 떨림 봇"),
        ("selenium_basic", "Selenium 기본 봇"),
        ("pyautogui", "PyAutoGUI 봇"),
        ("super_human", "초인간 흉내 봇"),
        ("neural_mimic", "신경망 흉내 봇"),
        ("hesitation", "망설임 봇"),
        ("overshoot", "오버슈트 봇"),
        ("curved_path", "곡선 경로 봇"),
        ("stutter", "끊김 봇"),
    ]
    
    TESTS_PER_TYPE = 100  # 각 유형별 100회
    
    print("=" * 70)
    print(f"Phase B 봇 탐지 스트레스 테스트 (각 유형 {TESTS_PER_TYPE}회, 총 {len(bot_types) * TESTS_PER_TYPE}회)")
    print("=" * 70)
    
    results = defaultdict(lambda: {"detected": 0, "passed": 0, "scores": []})
    total_detected = 0
    total_tests = 0
    
    for bot_type, name in bot_types:
        for i in range(TESTS_PER_TYPE):
            points = tester.generate_bot_drag(bot_type)
            result = tester.test_bot(points)
            
            score = result.get("score", 0)
            detected = not result.get("pass", True)
            
            results[bot_type]["scores"].append(score)
            if detected:
                results[bot_type]["detected"] += 1
                total_detected += 1
            else:
                results[bot_type]["passed"] += 1
            total_tests += 1
            
            if (i + 1) % 25 == 0:
                print(f"\r{name}: {i + 1}/{TESTS_PER_TYPE} 완료", end="", flush=True)
        
        detected = results[bot_type]["detected"]
        passed = results[bot_type]["passed"]
        avg_score = sum(results[bot_type]["scores"]) / len(results[bot_type]["scores"])
        min_score = min(results[bot_type]["scores"])
        max_score = max(results[bot_type]["scores"])
        rate = detected / TESTS_PER_TYPE * 100
        
        status = "✅" if rate >= 95 else "⚠️" if rate >= 80 else "❌"
        print(f"\r{name:20s}: {status} {rate:5.1f}% ({detected}/{TESTS_PER_TYPE}) | score: {avg_score:.3f} (min:{min_score:.3f}, max:{max_score:.3f})")
    
    print("\n" + "=" * 70)
    print("전체 결과")
    print("=" * 70)
    
    total_rate = total_detected / total_tests * 100
    print(f"전체 탐지율: {total_detected}/{total_tests} ({total_rate:.1f}%)")
    
    print("\n⚠️ 탐지율 95% 미만 봇:")
    for bot_type, name in bot_types:
        rate = results[bot_type]["detected"] / TESTS_PER_TYPE * 100
        if rate < 95:
            avg_score = sum(results[bot_type]["scores"]) / len(results[bot_type]["scores"])
            max_score = max(results[bot_type]["scores"])
            print(f"  - {name}: {rate:.1f}% (avg={avg_score:.3f}, max={max_score:.3f})")


if __name__ == "__main__":
    main()