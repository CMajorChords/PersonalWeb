# 提取用户上传的文本文件，并进行处理
import re
import base64
import docx
import markdown2
from PyPDF2 import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile


# 清理文本函数
def clean_text(text: str) -> str:
    """
    清理提取的文本，删除多余的空格和不需要的字符
    :param text: 要清理的文本
    :return: 清理后的文本
    """
    patterns = (
        r'\n+',  # 多个换行符
        r'\t+',  # 制表符
        r'\s+',  # 多个空格
        r'<[^>]+>',  # HTML标签
    )
    for pattern in patterns:
        text = re.sub(pattern, ' ', text)
    # 去除连续空格和特殊空白字符
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# 处理文件函数
def extract_text_from_basic_file(file: UploadedFile) -> str:
    """
    提取txt、py、m等无格式纯文本文件
    :param file: 文件本身，而非文件路径
    :return: 文本
    """
    text = file.read().decode("utf-8")
    return clean_text(text)


def extract_text_from_markdown(file: UploadedFile) -> str:
    """
    提取markdown文件
    :param file: 文件本身，而非文件路径
    :return: 文本
    """
    text = file.read().decode("utf-8")
    html = markdown2.markdown(text)
    return clean_text(html)


def extract_text_from_docx(file: UploadedFile) -> str:
    """
    提取docx文件
    :param file: 文件本身，而非文件路径
    :return: 文本
    """
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return clean_text(text)


def extract_text_from_pdf(file: UploadedFile) -> str:
    """
    提取pdf文件
    :param file: 文件本身，而非文件路径
    :return: 文本
    """
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return clean_text(text)


def extract_text(file: UploadedFile) -> (str, str):
    """
    提取文本文件
    :param file: 文件本身，而非文件路径
    :return: 文本和文件名称
    """
    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file)
    elif file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    else:
        text = extract_text_from_basic_file(file)
    return text, file.name


# def extract_text(files: Union[UploadedFile, list[UploadedFile]]) -> Tuple[list[str], list[str]]:
#     """
#     提取文本文件
#     :param files: 文件或文件列表，如果是文件列表，返回的文本是所有文件的文本拼接，每段文本前加上文件名
#     :return: 文本和文件名称
#     """
#     if isinstance(files, list):
#         texts = []
#         for file in files:
#             texts.append(rf"filename:\n{file.name}, text:\n{extract_text_from_single_file(file)}")
#         return texts, [file.name for file in files]
#     else:
#         return [extract_text_from_single_file(files)], [files.name]

def extract_image(file: UploadedFile) -> str:
    """
    提取图片文件
    :param file: 文件本身，而非文件路径
    :return: base64编码的图片和文件名称
    """
    return base64.b64encode(file.read()).decode("utf-8")
