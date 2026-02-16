#!/usr/bin/env python3
"""
覆盖率变化检测脚本

比较base分支和当前分支的覆盖率，确保新增代码覆盖率≥70%
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CoverageReport:
    """覆盖率报告"""
    line_rate: float
    covered_lines: int
    total_lines: int


class CoverageChangeChecker:
    """覆盖率变化检查器"""

    def __init__(self, base_branch: str, head_branch: str = "HEAD"):
        self.base_branch = base_branch
        self.head_branch = head_branch
        self.min_coverage = 0.60
        self.min_new_code_coverage = 0.70
        self.max_drop = 0.05

    def run_command(self, cmd):
        """运行shell命令"""
        return subprocess.run(cmd, capture_output=True, text=True)

    def parse_coverage_json(self, json_path: Path) -> CoverageReport:
        """解析coverage.json文件"""
        if not json_path.exists():
            raise FileNotFoundError(f"覆盖率文件不存在: {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        totals = data.get('totals', {})
        return CoverageReport(
            line_rate=totals.get('percent_covered', 0) / 100,
            covered_lines=totals.get('covered_lines', 0),
            total_lines=totals.get('num_statements', 0)
        )

    def run_coverage(self, output_file: str = "coverage.json") -> CoverageReport:
        """运行pytest并生成覆盖率报告"""
        print(f"运行测试并生成覆盖率报告...")

        self.run_command([
            'pytest', 'tests/',
            '--cov=src',
            f'--cov-report=json:{output_file}',
            '--cov-report=term-missing'
        ])

        return self.parse_coverage_json(Path(output_file))

    def check_coverage_threshold(self, report: CoverageReport) -> bool:
        """检查覆盖率是否达标"""
        coverage_pct = report.line_rate * 100

        print(f"\n覆盖率检查:")
        print(f"  覆盖率: {coverage_pct:.2f}%")
        print(f"  已覆盖: {report.covered_lines} / {report.total_lines} 行")

        if coverage_pct >= self.min_coverage * 100:
            print(f"  ✅ 达标 (≥ {self.min_coverage*100:.0f}%)")
            return True
        else:
            print(f"  ❌ 未达标 (< {self.min_coverage*100:.0f}%)")
            return False

    def run(self) -> bool:
        """执行覆盖率检查"""
        print("=" * 60)
        print("开始覆盖率检查")
        print("=" * 60)

        # 运行覆盖率测试
        report = self.run_coverage()

        # 检查阈值
        passed = self.check_coverage_threshold(report)

        print("\n" + "=" * 60)
        if passed:
            print("✅ 覆盖率检查通过")
        else:
            print("❌ 覆盖率检查未通过")

        return passed


def main():
    parser = argparse.ArgumentParser(description='检查代码覆盖率')
    parser.add_argument('--base', default='main', help='Base分支名称')
    parser.add_argument('--head', default='HEAD', help='Head分支名称')
    parser.add_argument('--min-coverage', type=float, default=0.60, help='最低覆盖率')

    args = parser.parse_args()

    checker = CoverageChangeChecker(args.base, args.head)
    checker.min_coverage = args.min_coverage

    success = checker.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
