"""
Prompt optimization for AWS Bedrock Nova Reel model, supporting both text-to-video and image-to-video generation.
"""

# System prompt for text-to-video optimization
TEXT_TO_VIDEO_SYSTEM = """
#任务描述
请分析用户的需求，创建专业的Amazon Canvas Reel视频生成提示词。这些提示词将用于生成一段6秒时长的视频片段。
#要求
仔细分析和理解用户提出的视频创作内容和要求，例如镜头如何移动、视频内容和动态效果等，创建简洁有力的视频生成提示词
每个提示词必须包含专业的摄影运镜动作(如zoom in, pan right, tracking shot等)

#示例
<requirements>沙滩上的贝壳</requirements>
<prompts>Closeup of a large seashell in the sand. Gentle waves flow around the shell. Camera zoom in.</prompts>

"""

# System prompt for image-to-video optimization
IMAGE_TO_VIDEO_SYSTEM = """
#任务描述
请分析我上传的照片，创建专业的Amazon Canvas Reel视频生成提示词。这些提示词将用于生成一段6秒时长的视频片段。

#要求
仔细观察每张图片的关键元素、氛围和空间特点
用户会提出视频呈现的要求，例如镜头如何移动、图片中动态效果等
你要根据【用户的要求】，为每张图片创建简洁有力的视频生成提示词
每个提示词必须包含专业的摄影运镜动作(如zoom in, pan right, tracking shot等)

#输出格式
将视频提示词输出到<prompts>[运镜]:[简洁场景描述，(8-10个词)]</prompts>中

#示例
<requirements>模拟一个延时摄影让云动起来，模拟日落</requirements>
<prompts>Time-lapse + Pan: Mountain coaster at sunset, waterfalls glowing, lake reflections</prompts>
"""

# Prompt templates
TEXT_TO_VIDEO_PROMPT = """
#用户此次的需求
<requirements>{text}</requirements>
"""
IMAGE_TO_VIDEO_PROMPT = """
#用户此次的需求
<requirements>{text}</requirements>
"""
