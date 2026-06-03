#!/usr/bin/env python3
"""
代码审查辅助工具
================

自动统计代码库质量指标，支持 CI 集成和本地开发使用。

用法:
    python scripts/code_review_helper.py              # 全量检查
    python scripts/code_review_helper.py --diff HEAD~1  # 增量检查
    python scripts/code_review_helper.py --fix           # 自动修复 (ruff)

作者: EFD-PINNs Team
日期: 2026-06-01
"""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


def _is_public(name: str) -> bool:
    """判断是否为公开函数/方法名。"""
    if name.startswith("__") and name.endswith("__"):
        return False  # dunder methods
    return not name.startswith("_")


def _has_docstring(node: ast.AST) -> bool:
    """检查 AST 节点是否有 docstring。"""
    return bool(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )


def count_functions_without_docstrings(path: Path) -> dict:
    """统计文件中缺少 docstring 的公开函数。

    Returns:
        {"total_public": int, "missing_docs": int, "functions": [str]}
    """
    result = {"total_public": 0, "missing_docs": 0, "functions": []}

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        result["functions"].append(f"PARSE_ERROR: {e}")
        return result

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_public(node.name):
                continue
            result["total_public"] += 1
            if not _has_docstring(node):
                result["missing_docs"] += 1
                result["functions"].append(f"{node.name}() L{node.lineno}")

    return result


def count_print_statements(path: Path) -> list:
    """检测文件中的 print() 调用 (应全部替换为 logger)。

    Returns:
        包含 print 调用的行信息列表
    """
    results = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return [f"PARSE_ERROR in {path}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            results.append(f"  print() at L{node.lineno}: {path.name}")
    return results


def count_test_assertions(path: Path) -> dict:
    """统计测试文件中的断言数量。

    Returns:
        {"assert_count": int, "test_functions": int}
    """
    result = {"assert_count": 0, "test_functions": 0}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            result["assert_count"] += 1
        elif isinstance(node, ast.Call):
            # Count self.assert*() (unittest) and pytest-style assert calls
            if isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith("assert"):
                    result["assert_count"] += 1
            elif isinstance(node.func, ast.Name) and node.func.id.startswith("assert"):
                result["assert_count"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            result["test_functions"] += 1
    return result


def count_lines(path: Path) -> int:
    """统计文件行数。"""
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return 0


def get_changed_files(diff_target: str = "HEAD~1") -> list:
    """获取 Git 变更的文件列表。"""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", diff_target],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [
            f.strip()
            for f in result.stdout.splitlines()
            if f.strip().endswith(".py") and not f.strip().startswith("outputs/")
        ]
    except FileNotFoundError:
        return []


def run_ruff_check(fix: bool = False) -> dict:
    """运行 ruff lint 检查。

    Returns:
        {"returncode": int, "stdout": str, "stderr": str}
    """
    cmd = ["uv", "run", "ruff", "check", str(SRC_DIR), str(TESTS_DIR)]
    if fix:
        cmd.append("--fix")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_ruff_format_check() -> dict:
    """运行 ruff format 检查。

    Returns:
        {"returncode": int, "stdout": str, "stderr": str}
    """
    cmd = ["uv", "run", "ruff", "format", "--check", str(SRC_DIR), str(TESTS_DIR)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_tests() -> dict:
    """运行测试套件。

    Returns:
        {"returncode": int, "stdout": str, "stderr": str}
    """
    cmd = ["uv", "run", "pytest", "-x", "-q", "--tb=short", str(TESTS_DIR)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def generate_report(changed_only: bool = False, diff_target: str = "HEAD~1") -> dict:
    """生成完整的代码质量报告。

    Returns:
        包含所有质量指标的字典
    """
    report: dict[str, Any] = {
        "timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True, check=False).stdout.strip(),
        "sections": [],
    }

    # 确定要检查的文件
    if changed_only:
        files = [PROJECT_ROOT / f for f in get_changed_files(diff_target)]
        if not files:
            report["sections"].append(
                {
                    "title": "Changed Files",
                    "status": "WARN",
                    "detail": "No changed Python files detected.",
                }
            )
            return report
    else:
        files = sorted(list(SRC_DIR.rglob("*.py")) + list(TESTS_DIR.rglob("*.py")))

    # 1. Docstring 覆盖率
    total_public = 0
    total_missing = 0
    doc_details = []
    for f in files:
        stats = count_functions_without_docstrings(f)
        total_public += stats["total_public"]
        total_missing += stats["missing_docs"]
        if stats["functions"]:
            doc_details.append(f"{f.relative_to(PROJECT_ROOT)}: {len(stats['functions'])} missing")

    coverage = 100 if total_public == 0 else round(100 * (total_public - total_missing) / total_public, 1)
    report["sections"].append(
        {
            "title": "Docstring Coverage",
            "status": "PASS" if coverage >= 80 else "WARN" if coverage >= 50 else "FAIL",
            "detail": f"{total_public - total_missing}/{total_public} ({coverage}%)",
            "files": doc_details[:20],  # top 20
        }
    )

    # 2. Print 语句检测
    print_found = []
    for f in files:
        print_found.extend(count_print_statements(f))

    report["sections"].append(
        {
            "title": "Print Statements",
            "status": "PASS" if len(print_found) == 0 else "FAIL",
            "detail": f"Found {len(print_found)} remaining print() calls",
            "instances": print_found,
        }
    )

    # 3. 测试断言统计
    total_asserts = 0
    total_test_fns = 0
    empty_tests = []
    for f in TESTS_DIR.glob("*.py"):
        stats = count_test_assertions(f)
        total_asserts += stats["assert_count"]
        total_test_fns += stats["test_functions"]
        if stats["test_functions"] > 0 and stats["assert_count"] == 0:
            empty_tests.append(f.name)

    report["sections"].append(
        {
            "title": "Test Coverage",
            "status": "PASS" if empty_tests == [] else "FAIL",
            "detail": f"{total_asserts} asserts across {total_test_fns} test functions",
            "empty_tests": empty_tests,
        }
    )

    # 4. 大文件 (>1000行)
    large_files = []
    for f in files:
        lines = count_lines(f)
        if lines > 1000:
            large_files.append(f"{f.relative_to(PROJECT_ROOT)} ({lines} lines)")

    report["sections"].append(
        {
            "title": "Large Files (>1000 lines)",
            "status": "PASS" if len(large_files) <= 3 else "WARN",
            "detail": f"{len(large_files)} files exceed 1000 lines",
            "files": large_files,
        }
    )

    # 5. Ruff 检查
    ruff = run_ruff_check(fix=False)
    report["sections"].append(
        {
            "title": "Ruff Lint",
            "status": "PASS" if ruff["returncode"] == 0 else "FAIL",
            "detail": "No issues found" if ruff["returncode"] == 0 else ruff["stdout"][:500],
        }
    )

    return report


def print_report(report: dict) -> None:
    """以可读格式打印报告。"""
    print("\n" + "=" * 60)
    print("  EFD-PINNs 代码质量报告")
    print(f"  {report['timestamp']}")
    print("=" * 60)

    fail_count = 0
    for section in report["sections"]:
        status_icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(section["status"], "[????]")
        if section["status"] == "FAIL":
            fail_count += 1
        print(f"\n{status_icon} {section['title']}: {section['detail']}")

        # 打印详情
        for key in ["files", "instances", "empty_tests"]:
            if section.get(key):
                for item in section[key][:15]:
                    print(f"      {item}")
                if len(section[key]) > 15:
                    print(f"      ... and {len(section[key]) - 15} more")

    print("\n" + "=" * 60)
    if fail_count == 0:
        print("  All checks passed!")
    else:
        print(f"  {fail_count} check(s) FAILED. Review above for details.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EFD-PINNs 代码审查辅助工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--diff",
        type=str,
        nargs="?",
        const="HEAD~1",
        help="仅检查 Git diff 范围内的变更文件 (默认: HEAD~1)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="自动运行 ruff --fix 修复可修复的问题",
    )
    parser.add_argument(
        "--format-check",
        action="store_true",
        help="运行 ruff format --check",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行测试套件",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出报告",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="全量检查 + fix + test",
    )

    args = parser.parse_args()

    # --all 模式：全量检查
    if args.all:
        print("Running ruff --fix ...")
        ruff_result = run_ruff_check(fix=True)
        if ruff_result["stdout"]:
            print(ruff_result["stdout"])
        if ruff_result["stderr"]:
            print(ruff_result["stderr"], file=sys.stderr)

        print("\nRunning ruff format --check ...")
        fmt_result = run_ruff_format_check()
        if fmt_result["stdout"]:
            print(fmt_result["stdout"])

        print("\nRunning tests ...")
        test_result = run_tests()
        print(test_result["stdout"])
        if test_result["stderr"]:
            print(test_result["stderr"], file=sys.stderr)

        print("\n" + "-" * 40)
        report = generate_report(changed_only=False)
        print_report(report)
        return

    # --fix 模式
    if args.fix:
        print("Running ruff --fix ...")
        result = run_ruff_check(fix=True)
        if result["stdout"]:
            print(result["stdout"])
        return

    # --format-check 模式
    if args.format_check:
        result = run_ruff_format_check()
        if result["returncode"] != 0:
            print("[FAIL] Format issues found:")
            print(result["stdout"])
            sys.exit(1)
        else:
            print("[PASS] All files formatted correctly.")
        return

    # --test 模式
    if args.test:
        result = run_tests()
        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"], file=sys.stderr)
        sys.exit(result["returncode"])

    # 默认：生成报告
    report = generate_report(
        changed_only=args.diff is not None,
        diff_target=args.diff or "HEAD~1",
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
