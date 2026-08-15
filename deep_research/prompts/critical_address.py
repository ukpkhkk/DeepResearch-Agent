#***********************************************
#      Filename: critical_address.py
#   Description: 紧急修复系统提示  
#***********************************************

CRITICAL_ADDRESS_PROMPT = """对抗团队在您的草稿中检测到以下问题：
{critique_text}

您需要在下一步中尝试去解决这些问题：
1.如果评论指出缺少信息，请调用“ConductResearch”查找并获取更多信息。
2.如果评论指出逻辑存在缺陷，请调用“think_tool”制定修复方案。
3.如果评论指出需要完善报告草稿，请调用“refine_draft_report”精炼草稿。
"""
