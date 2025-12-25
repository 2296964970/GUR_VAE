# SSH 远程代理配置指南

## 目标

让远程服务器（AutoDL）通过本机的代理上网，解决服务器无法访问 GitHub、HuggingFace 等网站的问题。

## 原理

```
┌──────────────┐       SSH 隧道        ┌──────────────┐
│  远程服务器   │  ◄─────────────────►  │  你的电脑    │
│   AutoDL     │                       │              │
│              │                       │    Clash     │
│ 127.0.0.1:10808 ──────转发到───────► 127.0.0.1:7890 │
└──────────────┘                       └──────────────┘
```

---

## 一次性配置

### 1. 本机 Windows：修改 SSH 配置

编辑 `C:\Users\汪宇\.ssh\config`：

```
Host connect.cqa1.seetacloud.com
  HostName connect.cqa1.seetacloud.com
  Port 49110
  User root
  RemoteForward 10808 127.0.0.1:7890
```

**说明：**
- `RemoteForward 10808 127.0.0.1:7890`：将远程服务器的 10808 端口转发到本机的 7890 端口（Clash 代理端口）
- 如果你的 Clash 端口不是 7890，需要修改为实际端口

### 2. 远程服务器：设置自动代理环境变量

SSH 连接到服务器后，执行以下命令（只需执行一次）：

```bash
echo 'export http_proxy=http://127.0.0.1:10808' >> ~/.bashrc
echo 'export https_proxy=http://127.0.0.1:10808' >> ~/.bashrc
source ~/.bashrc
```

---

### 3. 配置免密登录（可选但推荐）

在本机 PowerShell 执行：

```bash
# 生成密钥（如果没有）
ssh-keygen -t rsa -b 4096

# 复制公钥到服务器（需输入一次密码）
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh connect.cqa1.seetacloud.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

---

## 每次使用流程

| 步骤 | 位置 | 操作 |
|------|------|------|
| 1 | 本机 Windows | 确保 **Clash 正在运行** |
| 2 | 本机 Windows | 执行 `ssh connect.cqa1.seetacloud.com`（免密直接登录） |
| 3 | 远程服务器 | 连接成功后，代理自动生效 |

---

## 验证代理是否生效

在远程服务器上执行：

```bash
curl -x http://127.0.0.1:10808 -I https://www.google.com
```

如果返回 `HTTP/2 200`，说明配置成功。

---

## 注意事项

1. **必须从本机 SSH 连接**：代理转发只有通过 SSH 命令连接时才会生效
2. **本机 Clash 必须开着**：否则远程服务器无法通过代理上网
3. **VSCode Remote SSH**：如果使用 VSCode 连接，确保它读取了 `~/.ssh/config` 配置
4. **密码输入**：SSH 输入密码时不会显示任何字符，这是正常的安全设计

---

## 常见问题

### Q: curl 显示 Connection refused

**原因**：端口转发未生效

**解决**：
1. 确认本机 Clash 正在运行
2. 确认是从本机 PowerShell 用 `ssh` 命令连接的
3. 用 `ssh -v connect.cqa1.seetacloud.com` 查看调试信息，确认有 `Remote forward success`

### Q: 如何查看本机 Clash 端口

打开 Clash 软件 → 设置 → 查看「端口」或「Port」配置，常见默认值：
- HTTP 代理端口：7890
- SOCKS5 端口：7891
- 混合端口：7897

---

## 服务器信息

- 主机：`connect.cqa1.seetacloud.com`
- 端口：`49110`
- 用户：`root`
- 密码：`eiXSFZJJPLN8`
