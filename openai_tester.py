#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 公益站检测工具
用于测试 OpenAI 兼容接口的各种功能
"""

import json
import time
import sys
from typing import Optional, Dict, Any, List

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
except ImportError:
    print("请先安装 rich: pip install rich")
    sys.exit(1)

console = Console()


class OpenAITester:
    """OpenAI 接口测试器"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.results = {}
        self.all_test_history = []  # 保存所有测试历史，用于最终报告
        
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                      stream: bool = False, timeout: int = 30) -> requests.Response:
        """发送 HTTP 请求"""
        url = f"{self.base_url}{endpoint}"
        if method.upper() == "GET":
            return requests.get(url, headers=self.headers, timeout=timeout)
        elif method.upper() == "POST":
            return requests.post(url, headers=self.headers, json=data, 
                               stream=stream, timeout=timeout)
        raise ValueError(f"不支持的 HTTP 方法: {method}")
    
    def test_models_list(self) -> Dict[str, Any]:
        """测试获取模型列表"""
        console.print("\n[bold cyan]📋 测试模型列表...[/bold cyan]")
        result = {
            "success": False,
            "models": [],
            "error": None,
            "response_time": 0
        }
        
        try:
            start_time = time.time()
            response = self._make_request("GET", "/models")
            result["response_time"] = round(time.time() - start_time, 3)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                result["success"] = True
                result["models"] = [m.get("id", "unknown") for m in models]
                
                # 显示模型列表
                if result["models"]:
                    table = Table(title="支持的模型列表", show_header=True, header_style="bold magenta")
                    table.add_column("序号", style="cyan", width=6)
                    table.add_column("模型名称", style="green")
                    
                    for i, model in enumerate(result["models"], 1):
                        table.add_row(str(i), model)
                    
                    console.print(table)
                    console.print(f"[green]✅ 成功获取 {len(result['models'])} 个模型[/green]")
                else:
                    console.print("[yellow]⚠️ 模型列表为空[/yellow]")
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                console.print(f"[red]❌ 获取模型列表失败: {result['error']}[/red]")
                
        except requests.exceptions.Timeout:
            result["error"] = "请求超时"
            console.print("[red]❌ 请求超时[/red]")
        except requests.exceptions.ConnectionError as e:
            result["error"] = f"连接错误: {str(e)}"
            console.print(f"[red]❌ 连接错误: {e}[/red]")
        except Exception as e:
            result["error"] = str(e)
            console.print(f"[red]❌ 错误: {e}[/red]")
            
        self.results["models_list"] = result
        return result
    
    def test_chat_completion(self, model: Optional[str] = None) -> Dict[str, Any]:
        """测试基础对话功能"""
        console.print("\n[bold cyan]💬 测试基础对话...[/bold cyan]")
        result = {
            "success": False,
            "model_used": model,
            "response": None,
            "error": None,
            "response_time": 0
        }
        
        # 如果没有指定模型，尝试使用已获取的模型列表中的第一个
        if not model:
            models_result = self.results.get("models_list", {})
            if models_result.get("models"):
                # 优先选择 gpt 相关模型
                for m in models_result["models"]:
                    if "gpt" in m.lower():
                        model = m
                        break
                if not model:
                    model = models_result["models"][0]
            else:
                model = "gpt-3.5-turbo"  # 默认模型
        
        result["model_used"] = model
        console.print(f"[dim]使用模型: {model}[/dim]")
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "请用一句话介绍你自己"}
            ],
            "max_tokens": 100
        }
        
        try:
            start_time = time.time()
            response = self._make_request("POST", "/chat/completions", payload)
            result["response_time"] = round(time.time() - start_time, 3)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                result["success"] = True
                result["response"] = content
                console.print(f"[green]✅ 对话成功[/green]")
                console.print(Panel(content, title="AI 回复", border_style="green"))
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                console.print(f"[red]❌ 对话失败: {result['error']}[/red]")
                
        except requests.exceptions.Timeout:
            result["error"] = "请求超时"
            console.print("[red]❌ 请求超时[/red]")
        except Exception as e:
            result["error"] = str(e)
            console.print(f"[red]❌ 错误: {e}[/red]")
            
        self.results["chat_completion"] = result
        return result
    
    def test_stream_mode(self, model: Optional[str] = None) -> Dict[str, Any]:
        """测试流式输出 - 增强版，对比流式和非流式响应"""
        console.print("\n[bold cyan]🌊 测试 Stream 流式输出...[/bold cyan]")
        result = {
            "success": False,
            "model_used": model,
            "chunks_received": 0,
            "full_response": "",
            "error": None,
            "response_time": 0,
            "is_real_stream": False,  # 是否真正的流式
            "first_chunk_time": 0,    # 首个数据块时间
            "stream_quality": "unknown"  # 流式质量评估
        }
        
        # 选择模型
        if not model:
            models_result = self.results.get("models_list", {})
            if models_result.get("models"):
                for m in models_result["models"]:
                    if "gpt" in m.lower():
                        model = m
                        break
                if not model:
                    model = models_result["models"][0]
            else:
                model = "gpt-3.5-turbo"
        
        result["model_used"] = model
        console.print(f"[dim]使用模型: {model}[/dim]")
        
        # 测试用的消息
        test_message = "请从1数到10，每个数字单独输出"
        
        payload_stream = {
            "model": model,
            "messages": [
                {"role": "user", "content": test_message}
            ],
            "max_tokens": 100,
            "stream": True
        }
        
        try:
            start_time = time.time()
            first_chunk_time = None
            chunk_times = []  # 记录每个数据块的时间
            response = self._make_request("POST", "/chat/completions", payload_stream, stream=True, timeout=60)
            
            if response.status_code == 200:
                full_content = ""
                chunk_count = 0
                raw_lines = []
                has_done = False
                
                console.print("[dim]接收流式数据: [/dim]", end="")
                
                for line in response.iter_lines():
                    current_time = time.time()
                    
                    if line:
                        try:
                            line_str = line.decode('utf-8')
                        except:
                            line_str = str(line)
                        
                        raw_lines.append(line_str)
                        
                        # 处理标准 SSE 格式: data: {...}
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                has_done = True
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if first_chunk_time is None:
                                        first_chunk_time = current_time - start_time
                                    chunk_times.append(current_time - start_time)
                                    full_content += content
                                    chunk_count += 1
                                    console.print(f"[cyan]{content}[/cyan]", end="")
                            except json.JSONDecodeError:
                                pass
                        # 处理某些 API 直接返回 JSON 的情况
                        elif line_str.startswith("{"):
                            try:
                                data = json.loads(line_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if not content:
                                    content = data.get("content", "")
                                if not content:
                                    content = data.get("completion", "")
                                if content:
                                    if first_chunk_time is None:
                                        first_chunk_time = current_time - start_time
                                    chunk_times.append(current_time - start_time)
                                    full_content += content
                                    chunk_count += 1
                                    console.print(f"[cyan]{content}[/cyan]", end="")
                            except json.JSONDecodeError:
                                pass
                        elif line_str.startswith("event:"):
                            pass
                
                console.print()  # 换行
                total_time = time.time() - start_time
                result["response_time"] = round(total_time, 3)
                result["first_chunk_time"] = round(first_chunk_time, 3) if first_chunk_time else 0
                
                # 分析流式质量
                if chunk_count > 0:
                    result["chunks_received"] = chunk_count
                    result["full_response"] = full_content
                    
                    # 计算数据块之间的时间间隔
                    if len(chunk_times) > 1:
                        intervals = [chunk_times[i+1] - chunk_times[i] for i in range(len(chunk_times)-1)]
                        avg_interval = sum(intervals) / len(intervals)
                        
                        # 判断是否是真正的流式
                        # 真正的流式：数据块之间有明显的时间间隔
                        if avg_interval > 0.01 and chunk_count >= 3:  # 平均间隔 > 10ms 且至少3个块
                            result["is_real_stream"] = True
                            result["stream_quality"] = "excellent"
                            result["success"] = True
                            console.print(f"[green]✅ 真正的流式输出! 收到 {chunk_count} 个数据块[/green]")
                            console.print(f"[dim]   首字节时间: {result['first_chunk_time']}s, 平均间隔: {round(avg_interval*1000, 1)}ms[/dim]")
                        elif chunk_count >= 2:
                            result["is_real_stream"] = True
                            result["stream_quality"] = "good"
                            result["success"] = True
                            console.print(f"[green]✅ 流式输出支持! 收到 {chunk_count} 个数据块[/green]")
                        else:
                            result["stream_quality"] = "poor"
                            result["success"] = True
                            console.print(f"[yellow]⚠️ 流式输出可能是伪流式 (数据块太少)[/yellow]")
                    else:
                        # 只有一个数据块
                        result["stream_quality"] = "poor"
                        result["success"] = True
                        console.print(f"[yellow]⚠️ 只收到 1 个数据块，可能是伪流式[/yellow]")
                        
                elif full_content:
                    result["success"] = True
                    result["chunks_received"] = 1
                    result["full_response"] = full_content
                    result["stream_quality"] = "non-standard"
                    console.print(f"[yellow]⚠️ 非标准流式格式[/yellow]")
                else:
                    # 没有收到内容
                    result["success"] = False
                    
                    # 检查是否只有 [DONE]
                    if has_done and len(raw_lines) <= 2:
                        result["error"] = "API 返回空流式响应 (只有 [DONE])"
                        result["stream_quality"] = "not_supported"
                        console.print(f"[red]❌ 流式不支持: API 直接返回 [DONE]，没有实际数据[/red]")
                        console.print(f"[yellow]   这通常意味着该模型/API 不支持真正的流式输出[/yellow]")
                    else:
                        result["error"] = "未收到有效的流式数据"
                        result["stream_quality"] = "unknown"
                        console.print(f"[yellow]⚠️ 收到 0 个数据块[/yellow]")
                        if raw_lines:
                            console.print(f"[dim]原始响应 (前5行):[/dim]")
                            for i, raw_line in enumerate(raw_lines[:5]):
                                console.print(f"[dim]  {i+1}: {raw_line[:150]}{'...' if len(raw_line) > 150 else ''}[/dim]")
                    
                    # 尝试非流式请求作为对比
                    console.print(f"\n[dim]正在进行非流式对比测试...[/dim]")
                    try:
                        payload_non_stream = {
                            "model": model,
                            "messages": [{"role": "user", "content": "说'测试成功'"}],
                            "max_tokens": 20,
                            "stream": False
                        }
                        non_stream_start = time.time()
                        non_stream_resp = self._make_request("POST", "/chat/completions", payload_non_stream, timeout=30)
                        non_stream_time = time.time() - non_stream_start
                        
                        if non_stream_resp.status_code == 200:
                            data = non_stream_resp.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if content:
                                console.print(f"[green]   非流式请求成功 ({round(non_stream_time, 2)}s): {content[:50]}[/green]")
                                console.print(f"[yellow]   结论: API 可用，但流式模式可能不被该模型支持[/yellow]")
                            else:
                                console.print(f"[yellow]   非流式请求返回空内容[/yellow]")
                        else:
                            console.print(f"[red]   非流式请求也失败: HTTP {non_stream_resp.status_code}[/red]")
                    except Exception as e:
                        console.print(f"[red]   非流式对比测试失败: {e}[/red]")
                        
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                console.print(f"[red]❌ Stream 测试失败: {result['error']}[/red]")
                
        except requests.exceptions.Timeout:
            result["error"] = "请求超时"
            console.print("[red]❌ 请求超时[/red]")
        except Exception as e:
            result["error"] = str(e)
            console.print(f"[red]❌ 错误: {e}[/red]")
            
        self.results["stream_mode"] = result
        return result
    
    def test_function_calling(self, model: Optional[str] = None) -> Dict[str, Any]:
        """测试工具/函数调用"""
        console.print("\n[bold cyan]🔧 测试工具调用 (Function Calling)...[/bold cyan]")
        result = {
            "success": False,
            "model_used": model,
            "tool_called": False,
            "tool_name": None,
            "tool_arguments": None,
            "error": None,
            "response_time": 0
        }
        
        # 选择模型
        if not model:
            models_result = self.results.get("models_list", {})
            if models_result.get("models"):
                for m in models_result["models"]:
                    if "gpt" in m.lower():
                        model = m
                        break
                if not model:
                    model = models_result["models"][0]
            else:
                model = "gpt-3.5-turbo"
        
        result["model_used"] = model
        console.print(f"[dim]使用模型: {model}[/dim]")
        
        # 定义一个简单的工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称，如：北京、上海"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "北京今天天气怎么样？"}
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 200
        }
        
        try:
            start_time = time.time()
            response = self._make_request("POST", "/chat/completions", payload)
            result["response_time"] = round(time.time() - start_time, 3)
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    result["success"] = True
                    result["tool_called"] = True
                    result["tool_name"] = tool_calls[0].get("function", {}).get("name")
                    result["tool_arguments"] = tool_calls[0].get("function", {}).get("arguments")
                    
                    console.print(f"[green]✅ 工具调用支持![/green]")
                    console.print(f"[dim]调用的工具: {result['tool_name']}[/dim]")
                    console.print(f"[dim]参数: {result['tool_arguments']}[/dim]")
                else:
                    # 检查是否返回了普通回复（可能不支持工具调用）
                    content = message.get("content", "")
                    if content:
                        result["success"] = True
                        result["tool_called"] = False
                        console.print(f"[yellow]⚠️ 模型返回了普通回复，可能不支持工具调用[/yellow]")
                        console.print(f"[dim]回复: {content[:100]}...[/dim]")
                    else:
                        result["error"] = "未收到有效响应"
                        console.print("[red]❌ 未收到有效响应[/red]")
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                console.print(f"[red]❌ 工具调用测试失败: {result['error']}[/red]")
                
        except requests.exceptions.Timeout:
            result["error"] = "请求超时"
            console.print("[red]❌ 请求超时[/red]")
        except Exception as e:
            result["error"] = str(e)
            console.print(f"[red]❌ 错误: {e}[/red]")
            
        self.results["function_calling"] = result
        return result
    
    def test_embeddings(self, model: Optional[str] = None) -> Dict[str, Any]:
        """测试文本嵌入功能"""
        console.print("\n[bold cyan]📊 测试 Embeddings 文本嵌入...[/bold cyan]")
        result = {
            "success": False,
            "model_used": model,
            "dimensions": 0,
            "error": None,
            "response_time": 0
        }
        
        # 选择嵌入模型
        if not model:
            models_result = self.results.get("models_list", {})
            if models_result.get("models"):
                for m in models_result["models"]:
                    if "embed" in m.lower():
                        model = m
                        break
                if not model:
                    model = "text-embedding-ada-002"
            else:
                model = "text-embedding-ada-002"
        
        result["model_used"] = model
        console.print(f"[dim]使用模型: {model}[/dim]")
        
        payload = {
            "model": model,
            "input": "这是一个测试文本"
        }
        
        try:
            start_time = time.time()
            response = self._make_request("POST", "/embeddings", payload)
            result["response_time"] = round(time.time() - start_time, 3)
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("data", [{}])[0].get("embedding", [])
                result["success"] = True
                result["dimensions"] = len(embeddings)
                console.print(f"[green]✅ Embeddings 支持! 向量维度: {result['dimensions']}[/green]")
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                console.print(f"[red]❌ Embeddings 测试失败: {result['error']}[/red]")
                
        except requests.exceptions.Timeout:
            result["error"] = "请求超时"
            console.print("[red]❌ 请求超时[/red]")
        except Exception as e:
            result["error"] = str(e)
            console.print(f"[red]❌ 错误: {e}[/red]")
            
        self.results["embeddings"] = result
        return result
    
    def select_model(self, show_exit_option: bool = False) -> Optional[str]:
        """让用户选择模型
        
        Args:
            show_exit_option: 是否显示退出选项 (输入 0 退出)
        
        Returns:
            选择的模型名，如果用户选择退出则返回 "__EXIT__"
        """
        models_result = self.results.get("models_list", {})
        models = models_result.get("models", [])
        
        if not models:
            console.print("[yellow]⚠️ 没有可用的模型，将使用默认模型[/yellow]")
            return None
        
        console.print("\n[bold]请选择要测试的模型:[/bold]")
        if show_exit_option:
            console.print("[dim]输入序号选择模型，输入 0 退出程序[/dim]\n")
        else:
            console.print("[dim]输入序号选择模型，或直接输入模型名称[/dim]\n")
        
        # 分类显示模型
        chat_models = []
        embed_models = []
        other_models = []
        
        for m in models:
            m_lower = m.lower()
            if "embed" in m_lower:
                embed_models.append(m)
            elif any(x in m_lower for x in ["gpt", "claude", "llama", "qwen", "glm", "chat"]):
                chat_models.append(m)
            else:
                other_models.append(m)
        
        # 创建带分类的模型列表
        all_models_ordered = []
        
        if chat_models:
            console.print("[bold cyan]💬 对话模型:[/bold cyan]")
            for i, m in enumerate(chat_models, 1):
                console.print(f"  [cyan]{i}[/cyan]. {m}")
                all_models_ordered.append(m)
        
        offset = len(chat_models)
        if embed_models:
            console.print("[bold green]📊 嵌入模型:[/bold green]")
            for i, m in enumerate(embed_models, offset + 1):
                console.print(f"  [green]{i}[/green]. {m}")
                all_models_ordered.append(m)
        
        offset += len(embed_models)
        if other_models:
            console.print("[bold yellow]📦 其他模型:[/bold yellow]")
            for i, m in enumerate(other_models, offset + 1):
                console.print(f"  [yellow]{i}[/yellow]. {m}")
                all_models_ordered.append(m)
        
        console.print()
        
        while True:
            choice = console.input("[bold]请输入选择 (序号或模型名): [/bold]").strip()
            
            if not choice:
                # 默认选择第一个对话模型
                if chat_models:
                    selected = chat_models[0]
                    console.print(f"[dim]使用默认模型: {selected}[/dim]")
                    return selected
                elif all_models_ordered:
                    selected = all_models_ordered[0]
                    console.print(f"[dim]使用默认模型: {selected}[/dim]")
                    return selected
                return None
            
            # 尝试按序号选择
            try:
                idx = int(choice)
                # 检查是否输入 0 退出
                if show_exit_option and idx == 0:
                    return "__EXIT__"
                
                idx = idx - 1  # 转换为数组索引
                if 0 <= idx < len(all_models_ordered):
                    selected = all_models_ordered[idx]
                    console.print(f"[green]✓ 已选择模型: {selected}[/green]")
                    return selected
                else:
                    console.print("[red]序号超出范围，请重新输入[/red]")
            except ValueError:
                # 按名称选择
                if choice in models:
                    console.print(f"[green]✓ 已选择模型: {choice}[/green]")
                    return choice
                else:
                    # 模糊匹配
                    matches = [m for m in models if choice.lower() in m.lower()]
                    if len(matches) == 1:
                        console.print(f"[green]✓ 已选择模型: {matches[0]}[/green]")
                        return matches[0]
                    elif len(matches) > 1:
                        console.print(f"[yellow]找到多个匹配: {', '.join(matches[:5])}[/yellow]")
                        console.print("[yellow]请输入更精确的名称[/yellow]")
                    else:
                        console.print("[red]未找到匹配的模型，请重新输入[/red]")
    
    def select_embedding_model(self) -> Optional[str]:
        """选择嵌入模型"""
        models_result = self.results.get("models_list", {})
        models = models_result.get("models", [])
        
        # 查找嵌入模型
        embed_models = [m for m in models if "embed" in m.lower()]
        
        if embed_models:
            console.print(f"\n[bold]选择 Embeddings 测试模型:[/bold]")
            for i, m in enumerate(embed_models, 1):
                console.print(f"  [green]{i}[/green]. {m}")
            
            choice = console.input("[bold]请输入选择 (序号，直接回车使用第一个，输入 'skip' 跳过): [/bold]").strip()
            
            if choice.lower() == 'skip':
                return None
            
            if not choice:
                return embed_models[0]
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(embed_models):
                    return embed_models[idx]
            except ValueError:
                pass
            
            return embed_models[0]
        else:
            # 没有找到嵌入模型
            console.print(f"\n[yellow]⚠️ 未在模型列表中找到嵌入模型 (embedding)[/yellow]")
            choice = console.input("[bold]是否仍要测试 Embeddings? (输入模型名或直接回车跳过): [/bold]").strip()
            
            if not choice:
                return None  # 跳过测试
            
            return choice  # 使用用户输入的模型名
    
    def test_single_model(self, model: str) -> Dict[str, Any]:
        """测试单个模型的所有功能"""
        console.print(f"\n[bold blue]🚀 开始使用模型 [{model}] 进行功能测试...[/bold blue]")
        console.print("=" * 50)
        
        # 清空之前的测试结果（保留模型列表）
        models_list = self.results.get("models_list", {})
        self.results = {"models_list": models_list, "tested_model": model}
        
        # 使用选定的模型进行各项测试
        self.test_chat_completion(model=model)
        self.test_stream_mode(model=model)
        self.test_function_calling(model=model)
        
        # 生成测试报告
        self._print_summary()
        
        # 保存到测试历史
        self.all_test_history.append({
            "model": model,
            "chat": self.results.get("chat_completion", {}),
            "stream": self.results.get("stream_mode", {}),
            "tools": self.results.get("function_calling", {}),
            "embeddings": self.results.get("embeddings", {})
        })
        
        return self.results
    
    def _print_final_report(self):
        """打印最终对比报告"""
        if not self.all_test_history:
            return
        
        console.print("\n")
        console.print(Panel.fit(
            f"[bold]📊 测试总结报告[/bold]\n[dim]共测试 {len(self.all_test_history)} 个模型[/dim]",
            border_style="blue"
        ))
        
        # 流式质量描述映射
        stream_quality_desc = {
            "excellent": "真正流式",
            "good": "流式支持",
            "poor": "伪流式",
            "non-standard": "非标准",
            "not_supported": "不支持",
            "unknown": "未知"
        }
        
        # 创建对比表格
        table = Table(show_header=True, header_style="bold magenta", title="模型功能对比")
        table.add_column("模型", style="cyan", max_width=30)
        table.add_column("对话", justify="center")
        table.add_column("Stream", justify="center")
        table.add_column("工具调用", justify="center")
        
        for record in self.all_test_history:
            model_name = record["model"]
            # 截断过长的模型名
            if len(model_name) > 28:
                model_name = model_name[:25] + "..."
            
            # 对话状态
            chat = record.get("chat", {})
            chat_status = "[green]✅[/green]" if chat.get("success") else "[red]❌[/red]"
            
            # Stream 状态
            stream = record.get("stream", {})
            if stream.get("success"):
                quality = stream.get("stream_quality", "unknown")
                quality_text = stream_quality_desc.get(quality, quality)
                stream_status = f"[green]✅[/green] {quality_text}"
            else:
                stream_status = "[red]❌[/red]"
            
            # 工具调用状态
            tools = record.get("tools", {})
            if tools.get("success"):
                if tools.get("tool_called"):
                    tools_status = "[green]✅ 支持[/green]"
                else:
                    tools_status = "[yellow]⚠️ 未调用[/yellow]"
            else:
                tools_status = "[red]❌[/red]"
            
            table.add_row(model_name, chat_status, stream_status, tools_status)
        
        console.print(table)
        
        # 统计信息
        total = len(self.all_test_history)
        chat_pass = sum(1 for r in self.all_test_history if r.get("chat", {}).get("success"))
        stream_pass = sum(1 for r in self.all_test_history if r.get("stream", {}).get("success"))
        tools_pass = sum(1 for r in self.all_test_history if r.get("tools", {}).get("tool_called"))
        
        console.print(f"\n[bold]统计:[/bold]")
        console.print(f"  对话成功: {chat_pass}/{total}")
        console.print(f"  Stream 支持: {stream_pass}/{total}")
        console.print(f"  工具调用支持: {tools_pass}/{total}")
    
    def run_loop_mode(self) -> None:
        """循环测试模式 - 测试完成后返回模型列表，输入 0 退出"""
        console.print(Panel.fit(
            "[bold]OpenAI 公益站检测工具[/bold]\n"
            f"[dim]API 地址: {self.base_url}[/dim]",
            border_style="blue"
        ))
        
        # 1. 首先获取模型列表
        self.test_models_list()
        
        if not self.results.get("models_list", {}).get("success"):
            console.print("[red]无法获取模型列表，请检查 API 地址和 Key[/red]")
            return
        
        # 2. 循环测试
        test_count = 0
        while True:
            test_count += 1
            
            if test_count > 1:
                console.print("\n" + "=" * 50)
                console.print("[bold cyan]📋 返回模型列表[/bold cyan]")
            
            # 让用户选择模型（显示退出选项）
            selected_model = self.select_model(show_exit_option=True)
            
            # 检查是否退出
            if selected_model == "__EXIT__":
                # 打印最终对比报告
                if self.all_test_history:
                    self._print_final_report()
                console.print("\n[bold green]👋 感谢使用，再见！[/bold green]")
                break
            
            if not selected_model:
                console.print("[yellow]⚠️ 未选择模型，将使用默认模型 gpt-3.5-turbo[/yellow]")
                selected_model = "gpt-3.5-turbo"
            
            # 测试选定的模型
            self.test_single_model(selected_model)
            
            console.print("\n[bold green]✅ 本轮测试完成![/bold green]")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试（单次模式，兼容旧接口）"""
        console.print(Panel.fit(
            "[bold]OpenAI 公益站检测工具[/bold]\n"
            f"[dim]API 地址: {self.base_url}[/dim]",
            border_style="blue"
        ))
        
        # 1. 首先获取模型列表
        self.test_models_list()
        
        # 2. 让用户选择模型
        selected_model = self.select_model()
        
        if not selected_model:
            console.print("[yellow]⚠️ 未选择模型，将使用默认模型 gpt-3.5-turbo[/yellow]")
            selected_model = "gpt-3.5-turbo"
        
        # 3. 测试选定的模型
        self.test_single_model(selected_model)
        
        return self.results
    
    def _print_summary(self):
        """打印测试摘要"""
        tested_model = self.results.get("tested_model", "未知")
        
        console.print("\n")
        console.print(Panel.fit(f"[bold]📊 测试结果摘要[/bold]\n[dim]测试模型: {tested_model}[/dim]", border_style="blue"))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("测试项目", style="cyan")
        table.add_column("状态", justify="center")
        table.add_column("响应时间", justify="right")
        table.add_column("备注")
        
        # 流式质量描述映射
        stream_quality_desc = {
            "excellent": "真正流式",
            "good": "流式支持",
            "poor": "伪流式",
            "non-standard": "非标准格式",
            "not_supported": "不支持",
            "unknown": "未知"
        }
        
        def get_stream_note(r):
            chunks = r.get('chunks_received', 0)
            quality = r.get('stream_quality', 'unknown')
            quality_text = stream_quality_desc.get(quality, quality)
            if chunks > 0:
                return f"{chunks} 块 ({quality_text})"
            return quality_text
        
        test_items = [
            ("模型列表", "models_list", lambda r: f"{len(r.get('models', []))} 个模型"),
            ("基础对话", "chat_completion", lambda r: r.get("model_used", "")),
            ("Stream 模式", "stream_mode", get_stream_note),
            ("工具调用", "function_calling", lambda r: "已调用" if r.get("tool_called") else "未调用"),
        ]
        
        for name, key, note_func in test_items:
            result = self.results.get(key, {})
            # 处理跳过的测试
            if result.get("skipped"):
                status = "[yellow]⏭️ 跳过[/yellow]"
                note = "用户跳过"
            elif result.get("success"):
                status = "[green]✅ 通过[/green]"
                note = note_func(result)
            else:
                status = "[red]❌ 失败[/red]"
                error = result.get("error", "")
                note = (error[:30] + "...") if error and len(error) > 30 else error
            response_time = f"{result.get('response_time', 0)}s"
            table.add_row(name, status, response_time, note or "-")
        
        console.print(table)
        
        # 统计（过滤掉非字典类型的值，如 tested_model）
        test_results = [r for r in self.results.values() if isinstance(r, dict)]
        passed = sum(1 for r in test_results if r.get("success"))
        total = len(test_results)
        console.print(f"\n[bold]总计: {passed}/{total} 项测试通过[/bold]")


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold blue]🔍 OpenAI 公益站检测工具[/bold blue]\n"
        "[dim]测试 OpenAI 兼容接口的各种功能[/dim]",
        border_style="blue"
    ))
    
    # 获取用户输入
    console.print("\n[bold]请输入 API 信息:[/bold]")
    
    base_url = console.input("[cyan]API Base URL[/cyan] (如 https://api.openai.com/v1): ").strip()
    if not base_url:
        console.print("[red]错误: API Base URL 不能为空[/red]")
        return
    
    api_key = console.input("[cyan]API Key[/cyan]: ").strip()
    if not api_key:
        console.print("[red]错误: API Key 不能为空[/red]")
        return
    
    # 创建测试器并运行循环测试模式
    tester = OpenAITester(base_url, api_key)
    
    console.print("\n[bold yellow]开始测试...[/bold yellow]")
    console.print("=" * 50)
    
    # 使用循环测试模式
    tester.run_loop_mode()


if __name__ == "__main__":
    main()
