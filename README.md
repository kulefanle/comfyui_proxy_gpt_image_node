# Proxy GPT Image Generator for ComfyUI

这个自定义节点用于替代不稳定的 GPT image 代理节点，重点解决两类问题：

- 代理服务返回 `b64_json`、图片 URL、data URL、或 chat/completions 文本包装时，尽量自动解析成 ComfyUI `IMAGE`。
- 如果接口失败或没有真正返回图片，节点会直接报错，不再输出白色占位图误导排查。

## 安装

把 `comfyui_proxy_gpt_image_node` 整个目录放到：

```text
ComfyUI/custom_nodes/comfyui_proxy_gpt_image_node
```

然后重启 ComfyUI。

## 基础用法

在 ComfyUI 里添加节点：

```text
api/proxy -> Proxy GPT Image Generator
```

推荐先用最小参数测试：

```text
base_url: https://greenapi.ink
api_route: /v1/images/generations
model: gpt-image-2
prompt: a red apple on a wooden table, realistic photo
size: 1024x1024
quality: medium
background: opaque
output_format: png
```

输出连接：

```text
image -> PreviewImage
image_source -> PreviewAny
response_summary -> PreviewAny
```

## 如果代理走 chat/completions

如果你的代理服务说明要求请求：

```text
/v1/chat/completions
```

就把 `api_route` 改成：

```text
/v1/chat/completions
```

节点会用 OpenAI 风格的消息格式发送提示词和可选参考图。

## 排错

`HTTP 401`：

```text
API Key 和 base_url 不是同一家服务，或者账号没有模型权限。
```

`did not contain a usable image`：

```text
代理返回了成功 JSON，但里面没有 URL、data URL 或 b64_json。把 response_summary 接到 PreviewAny 查看实际返回格式。
```

仍然白图：

```text
优先检查 image_source。如果 image_source 是真实图片 URL，浏览器打开正常但 PreviewImage 白，说明下载/解码有问题。
如果 image_source 为空或是 auto，说明上游没有真正返回图片。
```

## 安全提醒

ComfyUI 工作流 JSON 会保存节点输入值。不要公开分享包含真实 API Key 的工作流。已经泄露的 Key 建议立即到服务商后台重置。
