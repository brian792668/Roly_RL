from tqdm import tqdm
import time

def print_status_bar(part1, part2, total_length=30):
    """
    用於終端的狀態條顯示，分成兩部分，每部分各有上限 0.5
    part1: 第一部分的值 (0 <= part1 <= 0.5)
    part2: 第二部分的值 (0 <= part2 <= 0.5)
    total_length: 狀態條的總長度
    """
    max_value = 0.5  # 每部分的上限值
    full_length = total_length  # 狀態條總長度

    # 計算每部分在條中的長度
    part1_length = int((part1 / max_value) * (full_length / 2))
    part2_length = int((part2 / max_value) * (full_length / 2))
    
    # 用等號 = 表示已達成的部分，用空白表示剩餘
    part1_bar = "=" * part1_length + " " * (full_length // 2 - part1_length)
    part2_bar = "=" * part2_length + " " * (full_length // 2 - part2_length)
    
    # 組合狀態條並輸出
    status_bar = f"|{part1_bar}{part2_bar}|"
    print(f"\r{status_bar} Part1: {part1:.2f}/0.5, Part2: {part2:.2f}/0.5", end="")

# 測試
if __name__ == "__main__":
    for i in range(51):  # 模擬部分1
        part1 = i / 100
        part2 = (50 - i) / 100
        print_status_bar(part1, part2)
        time.sleep(0.1)  # 模擬處理時間
    print()  # 結束後換行