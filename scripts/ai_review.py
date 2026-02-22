#!/usr/bin/env python3
"""
AI辅助代码Review脚本

检查高风险项并生成报告，不做最终裁判
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class RiskLevel(Enum):
    BLOCKER = "blocker"  # 必须修复
    MAJOR = "major"      # 建议修复
    MINOR = "minor"      # 可选改进


@dataclass
class ReviewIssue:
    """Review发现的问题"""
    file: str
    line: int
    risk_level: RiskLevel
    category: str
    message: str
    suggestion: str
    reproduction: Optional[str] = None


class AIReviewer:
    """AI辅助Reviewer"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.src_dir = self.project_root / "src"
        self.issues: List[ReviewIssue] = []

    def review(self, changed_files: List[str] = None) -> List[ReviewIssue]:
        """
        执行Review

        Args:
            changed_files: 只检查这些文件（用于PR diff检查）
                          如果为None，检查所有文件

        Returns:
            发现的问题列表
        """
        self.issues = []

        if changed_files:
            files_to_check = [self.project_root / f for f in changed_files
                             if f.endswith('.py') and (self.project_root / f).exists()]
        else:
            files_to_check = list(self.src_dir.rglob("*.py"))

        for file_path in files_to_check:
            self._check_file(file_path)

        return self.issues

    def _check_file(self, file_path: Path):
        """检查单个文件"""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            relative_path = file_path.relative_to(self.project_root)

            # 1. 检查敏感信息
            self._check_secrets(relative_path, lines)

            # 2. 检查前视偏差风险
            self._check_lookahead_bias(relative_path, lines)

            # 3. 检查除零保护
            self._check_division_safety(relative_path, lines)

            # 4. 检查SQL注入风险
            self._check_sql_injection(relative_path, lines)

            # 5. 检查异常处理
            self._check_exception_handling(relative_path, lines)

        except Exception as e:
            print(f"Error checking {file_path}: {e}")

    def _check_secrets(self, file_path: Path, lines: List[str]):
        """检查敏感信息泄露"""
        patterns = [
            (r'(token|api_key|secret|password)\s*=\s*["\'][^"\']+["\']',
             "硬编码的敏感信息"),
            (r'TUSHARE_TOKEN\s*=\s*["\'][^"\']+["\']',
             "Tushare Token硬编码"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # 检查是否是从环境变量读取
                    if 'os.environ' in line or 'getenv' in line:
                        continue
                    self.issues.append(ReviewIssue(
                        file=str(file_path),
                        line=i,
                        risk_level=RiskLevel.BLOCKER,
                        category="安全风险",
                        message=message,
                        suggestion="使用环境变量存储敏感信息: os.environ.get('TOKEN')",
                        reproduction="grep -rE '(token|password)\\s*=\\s*[\"\\']' src/"
                    ))

    def _check_lookahead_bias(self, file_path: Path, lines: List[str]):
        """检查前视偏差风险"""
        # 检查特征计算是否使用了shift
        in_feature_method = False

        for i, line in enumerate(lines, 1):
            # 检测是否在特征提取方法中
            if 'def _add_' in line or 'def extract' in line:
                in_feature_method = True

            if in_feature_method:
                # 检查是否直接使用close而没有shift
                if re.search(r'\["close"\][^.]|\.close[^.]', line):
                    if 'shift' not in line and 'pct_change' not in lines[max(0, i-3):i]:
                        # 这可能是一个问题，但如果是计算未来收益则没问题
                        if 'future' not in line.lower():
                            self.issues.append(ReviewIssue(
                                file=str(file_path),
                                line=i,
                                risk_level=RiskLevel.MAJOR,
                                category="前视偏差风险",
                                message="特征计算可能存在前视偏差",
                                suggestion="使用 df['close'].shift(1) 获取昨日收盘价",
                                reproduction="检查特征计算是否使用历史数据"
                            ))
                            # 避免重复报告
                            in_feature_method = False
                            break

    def _check_division_safety(self, file_path: Path, lines: List[str]):
        """检查除零保护"""
        division_patterns = [
            r'/\s*\w+\s*\)',           # x / y)
            r'/\s*\[',                  # / [
            r'pct_change\(\)',          # 可能产生inf
        ]

        for i, line in enumerate(lines, 1):
            for pattern in division_patterns:
                if re.search(pattern, line):
                    # 检查是否有保护措施
                    has_protection = any(x in line for x in [
                        'np.divide', 'where=', 'np.where',
                        'if', '!= 0', '> 0', 'fillna'
                    ])
                    if not has_protection:
                        self.issues.append(ReviewIssue(
                            file=str(file_path),
                            line=i,
                            risk_level=RiskLevel.MAJOR,
                            category="除零风险",
                            message="除法操作可能存在除零风险",
                            suggestion="使用 np.divide(..., where=denom!=0, out=默认值)",
                            reproduction="测试边界条件: pd.DataFrame({'a': [1], 'b': [0]})"
                        ))
                        break

    def _check_sql_injection(self, file_path: Path, lines: List[str]):
        """检查SQL注入风险"""
        for i, line in enumerate(lines, 1):
            if 'execute' in line.lower() and 'SELECT' in line.upper():
                # 检查是否使用参数化查询
                if '%' in line or '"+' in line or "'+" in line:
                    self.issues.append(ReviewIssue(
                        file=str(file_path),
                        line=i,
                        risk_level=RiskLevel.BLOCKER,
                        category="SQL注入风险",
                        message="SQL查询可能存在注入风险",
                        suggestion="使用参数化查询: cursor.execute(sql, params)",
                        reproduction="输入: '600000.SH'; DROP TABLE prices; --"
                    ))

    def _check_exception_handling(self, file_path: Path, lines: List[str]):
        """检查异常处理"""
        bare_except = False
        for i, line in enumerate(lines, 1):
            if 'except:' in line and 'except Exception' not in line:
                bare_except = True
                self.issues.append(ReviewIssue(
                    file=str(file_path),
                    line=i,
                    risk_level=RiskLevel.MINOR,
                    category="异常处理",
                    message="使用裸except可能捕获不应捕获的异常",
                    suggestion="使用 except Exception as e: 或更具体的异常类型"
                ))

    def generate_report(self) -> str:
        """生成Review报告"""
        if not self.issues:
            return "✅ 未发现高风险问题"

        # 按风险级别分组
        blockers = [i for i in self.issues if i.risk_level == RiskLevel.BLOCKER]
        majors = [i for i in self.issues if i.risk_level == RiskLevel.MAJOR]
        minors = [i for i in self.issues if i.risk_level == RiskLevel.MINOR]

        report = []
        report.append("=" * 60)
        report.append("AI Review Report")
        report.append("=" * 60)
        report.append("")

        if blockers:
            report.append(f"🚨 必须修复 (Blocker): {len(blockers)}")
            for issue in blockers:
                report.append(f"  {issue.file}:{issue.line}")
                report.append(f"    [{issue.category}] {issue.message}")
                report.append(f"    建议: {issue.suggestion}")
                if issue.reproduction:
                    report.append(f"    复现: {issue.reproduction}")
            report.append("")

        if majors:
            report.append(f"⚠️ 建议修复 (Major): {len(majors)}")
            for issue in majors:
                report.append(f"  {issue.file}:{issue.line}")
                report.append(f"    [{issue.category}] {issue.message}")
            report.append("")

        if minors:
            report.append(f"📝 可选改进 (Minor): {len(minors)}")
            for issue in minors:
                report.append(f"  {issue.file}:{issue.line}")
                report.append(f"    [{issue.category}] {issue.message}")
            report.append("")

        report.append("=" * 60)
        report.append("总结:")
        report.append(f"  Blocker: {len(blockers)} | Major: {len(majors)} | Minor: {len(minors)}")
        report.append("")

        if blockers:
            report.append("❌ 存在必须修复的问题，建议修复后再合并")
        elif majors:
            report.append("⚠️ 存在建议修复的问题，请评估后决定")
        else:
            report.append("✅ 只存在可选改进项")

        return "\n".join(report)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="AI辅助代码Review")
    parser.add_argument("--files", nargs="*", help="只检查指定文件")
    parser.add_argument("--diff", action="store_true", help="只检查git diff中的文件")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")

    args = parser.parse_args()

    reviewer = AIReviewer()

    # 确定要检查的文件
    if args.diff:
        # 获取git diff中的文件
        result = subprocess.run(
            ["git", "diff", "--name-only", "main"],
            capture_output=True, text=True
        )
        changed_files = result.stdout.strip().split('\n')
        changed_files = [f for f in changed_files if f and f.endswith('.py')]
        print(f"检查 {len(changed_files)} 个变更文件...")
        issues = reviewer.review(changed_files)
    elif args.files:
        issues = reviewer.review(args.files)
    else:
        print("检查所有源文件...")
        issues = reviewer.review()

    # 输出报告
    if args.output == "json":
        import json
        output = {
            "issues": [
                {
                    "file": i.file,
                    "line": i.line,
                    "risk_level": i.risk_level.value,
                    "category": i.category,
                    "message": i.message,
                    "suggestion": i.suggestion,
                }
                for i in issues
            ],
            "summary": {
                "blockers": len([i for i in issues if i.risk_level == RiskLevel.BLOCKER]),
                "majors": len([i for i in issues if i.risk_level == RiskLevel.MAJOR]),
                "minors": len([i for i in issues if i.risk_level == RiskLevel.MINOR]),
            }
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(reviewer.generate_report())

    # 返回码：有blocker则返回1
    if any(i.risk_level == RiskLevel.BLOCKER for i in issues):
        sys.exit(1)


if __name__ == "__main__":
    main()
