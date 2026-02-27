# serve_policy.py 详细说明

## 概述

`serve_policy.py` 是一个策略服务器脚本，它启动一个 WebSocket 服务器来提供策略推理服务。根据配置，它可能生成以下内容：

---

## 1. WebSocket 服务器（主要功能）

### 功能
- 在指定端口（默认 8000）上启动 WebSocket 服务器
- 接收客户端的观察数据（observations）
- 返回策略推理的动作（actions）

### 生成的内容
**服务器本身不生成文件**，它只是提供实时推理服务。

### 服务器行为
1. **启动时**：
   - 加载模型检查点（checkpoint）
   - 初始化策略
   - 在 `0.0.0.0:8000` 端口监听连接

2. **运行时**：
   - 接收客户端连接
   - 发送策略元数据（metadata）给客户端
   - 接收观察数据 → 推理 → 返回动作

3. **日志输出**：
   ```
   INFO: Creating server (host: lgn9, ip: 127.0.1.1)
   INFO: server listening on 0.0.0.0:8000
   INFO: Connection from (IP, port) opened
   INFO: Connection from (IP, port) closed
   ```

---

## 2. Policy Records（可选，使用 --record 参数时）

### 启用方式
```bash
uv run scripts/serve_policy.py --env LIBERO --record
```

### 生成位置
- 目录：`policy_records/`（在运行脚本的当前目录下）
- 文件格式：`step_0.npy`, `step_1.npy`, `step_2.npy`, ...

### 每个记录文件包含的内容

每个 `.npy` 文件是一个扁平化的字典，包含：

#### 输入数据（inputs/）
- `inputs/observation/image`: (224, 224, 3) uint8
  - 主视角图像（RGB）
  
- `inputs/observation/wrist_image`: (224, 224, 3) uint8
  - 手腕相机图像（RGB）
  
- `inputs/observation/state`: (8,) float64
  - 机器人状态向量，包含：
    - 末端执行器位置（3维）
    - 末端执行器姿态（3维，轴角表示）
    - 夹爪位置（1维）
    - 其他状态（1维）
  
- `inputs/prompt`: str
  - 任务描述文本（例如："pick up the red cup"）

#### 输出数据（outputs/）
- `outputs/actions`: (10, 7) float64
  - 策略预测的动作序列
  - 10 步未来动作，每步 7 维（6维位置+姿态 + 1维夹爪）
  
- `outputs/policy_timing/infer_ms`: float
  - 策略推理时间（毫秒）

### 数据结构示例
```python
{
    "inputs/observation/image": np.ndarray,      # (224, 224, 3) uint8
    "inputs/observation/wrist_image": np.ndarray, # (224, 224, 3) uint8
    "inputs/observation/state": np.ndarray,       # (8,) float64
    "inputs/prompt": str,                         # 任务描述
    "outputs/actions": np.ndarray,                # (10, 7) float64
    "outputs/policy_timing/infer_ms": float       # 推理时间
}
```

### 用途
- **调试**：检查策略的输入输出
- **分析**：分析策略行为
- **可视化**：查看输入图像和预测动作
- **回放**：重现策略决策过程

---

## 3. 策略元数据（Metadata）

### 内容
服务器在客户端连接时发送策略元数据，包含：
- 模型类型（model_type）
- 配置信息（config）
- 其他策略相关信息

### 获取方式
客户端连接时会自动接收元数据：
```python
client = WebsocketClientPolicy(host="localhost", port=8000)
metadata = client.get_server_metadata()
print(metadata)
```

---

## 4. 服务器时序信息（Server Timing）

### 内容
每次推理返回的动作字典中包含时序信息：
```python
{
    "actions": np.ndarray,  # 动作序列
    "server_timing": {
        "infer_ms": float,        # 服务器端推理时间（毫秒）
        "prev_total_ms": float    # 上一次请求的总处理时间（毫秒）
    },
    "policy_timing": {
        "infer_ms": float         # 策略内部推理时间（毫秒）
    }
}
```

### 用途
- 性能分析
- 延迟监控
- 优化参考

---

## 5. 健康检查端点

### 功能
服务器提供 HTTP 健康检查端点：
- URL: `http://localhost:8000/healthz`
- 返回: `OK\n`（如果服务器正常运行）

### 用途
- 检查服务器是否运行
- 监控服务器状态
- 负载均衡器健康检查

---

## 总结

### 生成的文件/数据

| 类型 | 位置 | 格式 | 何时生成 |
|------|------|------|----------|
| Policy Records | `policy_records/step_*.npy` | NumPy 数组 | 使用 `--record` 参数时 |
| 服务器日志 | 终端输出 | 文本 | 始终 |
| WebSocket 服务 | 内存中 | 实时数据流 | 始终 |
| 元数据 | 通过 WebSocket 发送 | 字典 | 客户端连接时 |

### 关键点

1. **服务器本身不生成持久化文件**（除非使用 `--record`）
2. **所有数据通过 WebSocket 实时传输**
3. **记录文件用于调试和分析**，不是必需的
4. **时序信息帮助性能分析**

### 使用建议

- **正常使用**：不需要 `--record`，直接运行服务器
- **调试分析**：使用 `--record` 保存输入输出
- **性能测试**：查看 `server_timing` 和 `policy_timing`
- **监控**：使用健康检查端点检查服务器状态
