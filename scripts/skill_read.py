#!/usr/bin/env python3
"""
skill_read.py — 读取/搜索 .skill 文件内容（zip格式，不可直接 grep）

用法：
  python scripts/skill_read.py morning-dashboard.skill          # 读全部内容
  python scripts/skill_read.py morning-dashboard.skill SKILL.md # 读指定文件
  python scripts/skill_read.py --grep "关键词"                   # 搜索所有 skill

示例：
  python scripts/skill_read.py --grep "KOL"
  python scripts/skill_read.py seo-expert.skill
"""
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).parent.parent


def read_skill(skill_path: Path, target_file: str = None):
    with zipfile.ZipFile(skill_path) as z:
        names = z.namelist()
        if target_file:
            names = [n for n in names if n == target_file]
        for name in names:
            content = z.read(name).decode("utf-8", errors="replace")
            print(f"\n{'='*60}")
            print(f"  {skill_path.name} / {name}")
            print(f"{'='*60}")
            print(content)


def grep_skills(keyword: str):
    found = False
    for skill_file in sorted(BASE.glob("*.skill")):
        try:
            with zipfile.ZipFile(skill_file) as z:
                for name in z.namelist():
                    content = z.read(name).decode("utf-8", errors="replace")
                    matches = [
                        (i + 1, line)
                        for i, line in enumerate(content.splitlines())
                        if keyword.lower() in line.lower()
                    ]
                    if matches:
                        found = True
                        print(f"\n{skill_file.name} / {name}:")
                        for lineno, line in matches:
                            print(f"  {lineno:3}: {line.strip()}")
        except Exception as e:
            print(f"  跳过 {skill_file.name}: {e}")
    if not found:
        print(f"未找到包含 '{keyword}' 的内容")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    if args[0] == "--grep":
        if len(args) < 2:
            print("用法: python scripts/skill_read.py --grep <关键词>")
            sys.exit(1)
        grep_skills(args[1])
    else:
        skill_path = BASE / args[0]
        if not skill_path.exists():
            print(f"文件不存在: {skill_path}")
            sys.exit(1)
        target = args[1] if len(args) > 1 else None
        read_skill(skill_path, target)
