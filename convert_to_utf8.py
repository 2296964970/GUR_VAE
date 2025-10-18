"""
将项目中所有文本文件转换为UTF-8编码
"""
import os
import glob
import chardet

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def convert_to_utf8(file_path):
    """将文件转换为UTF-8编码"""
    try:
        # 检测原始编码
        original_encoding = detect_encoding(file_path)

        if original_encoding is None:
            print(f"[SKIP] 无法检测编码: {file_path}")
            return False

        # 如果已经是UTF-8（无BOM），跳过
        if original_encoding.lower() in ['utf-8', 'ascii']:
            print(f"[OK] 已是UTF-8(无BOM): {file_path}")
            return True

        # 读取文件内容（包括UTF-8-SIG的情况）
        with open(file_path, 'r', encoding=original_encoding, errors='ignore') as f:
            content = f.read()

        # 写回UTF-8编码（无BOM）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[CONVERTED] {original_encoding} -> UTF-8(无BOM): {file_path}")
        return True

    except Exception as e:
        print(f"[ERROR] 转换失败 {file_path}: {e}")
        return False

def main():
    """主函数"""
    # 定义需要转换的文件模式
    patterns = [
        '**/*.py',
        '**/*.md',
        '*.txt',
        '*.toml',
        '.gitignore'
    ]

    files_to_convert = set()

    # 收集所有文件
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        files_to_convert.update(files)

    # 排除某些目录
    excluded_dirs = ['__pycache__', '.git', 'venv', 'env', '.venv']
    files_to_convert = [
        f for f in files_to_convert
        if not any(excluded in f for excluded in excluded_dirs)
    ]

    print(f"找到 {len(files_to_convert)} 个文件需要处理\n")

    # 转换每个文件
    success_count = 0
    for file_path in sorted(files_to_convert):
        if convert_to_utf8(file_path):
            success_count += 1

    print(f"\n完成！成功处理 {success_count}/{len(files_to_convert)} 个文件")

if __name__ == '__main__':
    main()
