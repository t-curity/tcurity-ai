# test_phase_b_human.py
import requests
import json
import glob
import os

AI_SERVER = "http://localhost:9000"

def test_human_data():
    # human 폴더와 human_pred 폴더 모두 테스트
    human_dirs = [
        "/home/ubuntu/tcurity-ai/data/phase_b/human",
        "/home/ubuntu/tcurity-ai/data/phase_b/human_pred",
    ]
    
    results = {"passed": 0, "failed": 0, "errors": 0, "scores": []}
    failed_files = []
    
    all_files = []
    for d in human_dirs:
        all_files.extend(glob.glob(f"{d}/**/*.json", recursive=True))
        all_files.extend(glob.glob(f"{d}/*.json"))
    
    all_files = list(set(all_files))  # 중복 제거
    total = len(all_files)
    
    print(f"총 사람 데이터: {total}개")
    print("=" * 60)
    
    for i, filepath in enumerate(all_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # behavior wrapper 처리
            if "behavior" in data:
                payload = data["behavior"]
            else:
                payload = data
            
            resp = requests.post(
                f"{AI_SERVER}/test/phase_b",
                json=payload,
                timeout=10
            )
            result = resp.json()
            
            score = result.get("score", 0)
            passed = result.get("pass", False)
            
            results["scores"].append(score)
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                failed_files.append({
                    "file": filepath,
                    "score": score
                })
        
        except Exception as e:
            results["errors"] += 1
        
        if (i + 1) % 100 == 0:
            print(f"\r진행: {i + 1}/{total} ({(i+1)/total*100:.1f}%)", end="", flush=True)
    
    print(f"\r진행: {total}/{total} (100.0%)")
    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)
    
    pass_rate = results["passed"] / (total - results["errors"]) * 100 if (total - results["errors"]) > 0 else 0
    
    print(f"통과: {results['passed']}")
    print(f"실패: {results['failed']}")
    print(f"에러: {results['errors']}")
    print(f"통과율: {pass_rate:.2f}%")
    
    if results["scores"]:
        avg_score = sum(results["scores"]) / len(results["scores"])
        min_score = min(results["scores"])
        max_score = max(results["scores"])
        print(f"\nScore 통계:")
        print(f"  평균: {avg_score:.4f}")
        print(f"  최소: {min_score:.4f}")
        print(f"  최대: {max_score:.4f}")
    
    if failed_files:
        print(f"\n⚠️ 실패한 파일 (상위 10개):")
        failed_files.sort(key=lambda x: x["score"])
        for f in failed_files[:10]:
            print(f"  - {os.path.basename(f['file'])}: score={f['score']:.4f}")


if __name__ == "__main__":
    test_human_data()