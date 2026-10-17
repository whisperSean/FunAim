# FunAim · 人物锁定趣味演示 App（iOS）

纯本地运行：摄像头采集 → Vision 人物检测 → 矩形框标注 → 准星瞄准射线 → 可拖拽悬浮控制窗。
**全程设备端推理，无任何网络请求。**

## 需求实现对照

| # | 需求 | 实现位置 |
|---|------|----------|
| 1 | 启动第一时间申请摄像头权限 + 失败引导跳转 | `Core/PermissionManager.swift`、`Views/ContentView.swift`（launchView / deniedView，含"前往系统设置"与"重新检测"） |
| 2 | 本地人物检测 + 实时矩形框标记 | `Core/PersonDetector.swift`（VNDetectHumanRectanglesRequest + 跨帧 ID 跟踪）、`Views/OverlayView.swift` |
| 3 | 悬浮窗口 + 叠加绘制射线等元素 | `Views/Components/LiquidGlass.swift`（灵动岛式菜单）、`Views/OverlayView.swift`（顶部居中射线/瞄准线/准星/置信度标签） |
| 4 | 自动瞄准锁定人物目标 | `Core/AimEngine.swift`（最近邻目标选择 + 粘滞锁定 + 指数平滑追踪） |
| 5 | 完整设置界面 | `Views/SettingsView.swift`（运行模式/灵敏度/瞄准开关与参数/边框样式/颜色/HUD 开关/运行状态） |
| 6 | 全部本地推理，无云端依赖 | Vision 框架设备端执行，工程无任何网络代码 |
| 7 | 60fps 流畅运行 | `Core/CameraService.swift`（自动选择 720p@60fps 格式；预览层满帧渲染与推理管线解耦；晚期帧丢弃策略） |
| 8 | iOS 15+ / iPhone 全系列 | 部署目标 iOS 15.0，仅 Portrait，SwiftUI Canvas（iOS 15 API） |

## 工程结构

```
FunAim/
├── project.yml                 # XcodeGen 工程描述（含 Info.plist 全部键值，构建时自动生成）
└── FunAim/
    ├── App/FunAimApp.swift     # 入口
    ├── Core/
    │   ├── PermissionManager.swift  # 权限申请/刷新/跳转系统设置
    │   ├── CameraService.swift      # 采集会话、60fps 格式选择、FPS 统计、帧分发
    │   ├── PersonDetector.swift     # 本地人物检测 + 目标跟踪
    │   ├── AimEngine.swift          # 目标选择 + 准星平滑追踪
    │   └── CoordinateMapper.swift   # Vision 坐标 → 视图坐标（等比填充逆映射）
    ├── Models/AppSettings.swift     # 设置中心 + 持久化 + 颜色工具
    └── Views/
        ├── ContentView.swift        # 权限门控 + 横屏主界面组装
        ├── CameraPreviewView.swift  # AVCaptureVideoPreviewLayer 封装
        ├── OverlayView.swift        # Canvas 叠加绘制（方框/射线/准星）
        ├── SettingsView.swift       # 设置面板（灵动岛展开层）
        └── Components/LiquidGlass.swift  # 液态玻璃组件库 · 动效令牌 · 灵动岛式状态胶囊
```

## 没有 Mac 怎么构建

本项目为原生 iOS（Swift/SwiftUI + AVFoundation + Vision/CoreML），Apple 工具链硬限制：编译必须 macOS + Xcode，Windows 无法本地编译（swift.org 的 Windows 工具链不含 UIKit/SwiftUI；换 Flutter/RN 也绕不开 macOS 出包）。可行路径按成本排序：

| 路径 | 成本 | 说明 |
|---|---|---|
| **GitHub Actions 云构建（已配好，推荐）** | 免费（公开仓库） | push 自动编译 + 自动导出未签名 IPA，Actions 页面直接下载，无需 Mac、无需签名证书 |
| 云 Mac 按需租 | 约 $1–2/小时 | MacinCloud / MacStadium / AWS EC2 Mac 实例，远程桌面装 Xcode（App Store 免费）本地构建 |
| 二手 Mac mini | 一次性 | 长期开发最省心，M1 二手即可 |

### 云端自动构建流水线（已配置）

仓库自带 `.github/workflows/ios-build.yml`，运行记录：https://github.com/whisperSean/FunAim/actions

**触发条件**：push 改动 `FunAim/**`、`project.yml`、`Info.plist` 或工作流自身；也可在 Actions 页面手动 `Run workflow`。

**两个并行任务**（macOS runner，约 1 分钟跑完）：

| 任务 | 内容 | 产物（Artifacts） |
|---|---|---|
| build | XcodeGen 生成工程 → 模拟器编译验证 | `FunAim-simulator-app` |
| ipa | xcodebuild archive（免签名）→ Payload 打包 | **`FunAim-unsigned-ipa`**（保留 30 天） |

**下载 IPA 的两种方式**：

1. 网页：Actions → 点进绿色勾的运行 → 页面底部 **Artifacts** → 下载 `FunAim-unsigned-ipa`（需登录 GitHub，zip 解压得 `FunAim-unsigned.ipa`）；
2. 命令行（本机已登录 gh CLI）：

```bash
gh run download -R whisperSean/FunAim -n FunAim-unsigned-ipa -D ./IPA成品
# gh 自动解压，得到 FunAim-unsigned.ipa
```

**IPA 装机（Windows → iPhone）**：

1. 电脑装 [Sideloadly](https://sideloadly.io)（免费）；
2. USB 连接 iPhone → 拖入 `FunAim-unsigned.ipa` → 输入 Apple ID → Start；
3. 手机上：设置 → 通用 → VPN 与设备管理 → 信任 Apple ID 开发者证书；
4. 免费 Apple ID 签名 **7 天有效**，到期用 Sideloadly 重签一次；$99/年 Apple Developer Program 可长期签名 + TestFlight 分发。

**流水线踩坑记录（已固化，勿回退）**：

- XcodeGen 固定 2.43.0（GitHub Release 直装，不走 brew）：brew 最新版写出的工程格式，macos-14 runner 的 Xcode 15.4 读不了（报 `Unable to read project`）；
- 推送 `.github/workflows/` 下改动需要令牌带 `workflow` scope：`gh auth login --scopes repo,workflow`（设备码浏览器授权）；
- `CGImagePropertyOrientation` 的成员是 `.up/.down/.left/.right`，不是 `.top`；
- 模拟器没有摄像头，云构建只能验证"可编译"，功能验证必须真机。

### 云 Mac 手动出未签名 IPA（备选）

```bash
xcodebuild archive -project FunAim.xcodeproj -scheme FunAim \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO \
  -archivePath FunAim.xcarchive
cd FunAim.xcarchive/Products/Applications
mkdir Payload && cp -r FunAim.app Payload/ && zip -r FunAim.ipa Payload
```

通用注意：
- **真机运行必须签名**：免费 Apple ID（7 天）或 Apple Developer Program（$99/年）；
- 工程内代码对这些路径零适配成本（XcodeGen 工程 + 无第三方依赖）。

## 构建（需 Mac + Xcode 13+；Windows 无法编译 iOS 应用）

### 方式 A：XcodeGen（推荐）

```bash
brew install xcodegen
cd FunAim
xcodegen                # 生成 FunAim.xcodeproj
open FunAim.xcodeproj
```

在 Xcode 中：选中真机 → Signing & Capabilities 勾选你的 Team → `Cmd+R` 运行。

### 方式 B：手动建工程

1. Xcode → Create New Project → iOS App，命名 `FunAim`，Interface 选 SwiftUI，最低版本 iOS 15.0；
2. 删除模板 `ContentView.swift`；
3. 把 `FunAim/FunAim/` 下全部 `.swift` 拖入工程（勾选 Copy items if needed）；
4. Info.plist 必须包含 `NSCameraUsageDescription`（缺失会导致启动请求权限时直接闪退；云构建由 project.yml 的 `info.properties` 自动生成）；
5. 勾选 Portrait 单方向，iPhone 目标。

## 测试注意

- **必须真机运行**：模拟器没有摄像头，无法出画面；
- 首次启动立即弹出权限申请；若拒绝后可在系统设置中重新开启，返回 App 点"我已开启，重新检测"；
- 旧机型若觉卡顿，设置 → 运行模式选"流畅"（每 3 帧推理一次，预览仍满帧）；
- 若需更换自定义 CoreML 模型（如 YOLO），只需替换 `PersonDetector` 内的请求实现，输出归一化框即可无缝接入后续管线。

## 性能设计要点

- 预览层（AVCaptureVideoPreviewLayer）独立满帧渲染，AI 推理在独立串行队列按帧间隔执行，二者互不阻塞；
- `alwaysDiscardsLateVideoFrames = true` 防止推理过慢导致画面滞后；
- 瞄准准星采用指数平滑 + 30Hz 更新 + Canvas 线性补间动画，视觉上保持 60fps 连贯；
- 画面坐标通过等比填充逆映射（CoordinateMapper）与检测框严格对齐，无偏移。

## 检测引擎（YOLO11n · 单一路径）

统一使用 CI 自动导出打包的 **YOLO11n CoreML 模型**：仅识别人物（person 类过滤 + IoU NMS），
VNCoreMLRequest 自动调度 ANE/GPU；模型导出失败会直接导致构建失败（不做降级兜底，保证路径唯一可靠）。
模型缺失时检测返回空结果。

### YOLO11n（推荐，Ultralytics 官方 CoreML 导出）

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=coreml imgsz=640   # 生成 yolo11n.mlmodel
```

1. 把 `yolo11n.mlmodel` 拖入 Xcode 工程，勾选 FunAim target；
2. 重新运行：`PersonDetector` 发现 Bundle 内存在该模型即自动切换到 `VNCoreMLRequest`（ANE 加速），并自动完成 person 类别过滤与 IoU NMS；
3. 换其他模型只需改 `PersonDetector.coreMLModelName`（如 `yolov8n`）。

备选来源：Hugging Face 上已转换好的 `yolov8n` CoreML 权重（搜 "yolov8n coreml"）；或 Google MediaPipe / TFLite SSD-MobileNet（需自接采集与渲染管线，改动更大，不推荐本项目场景）。

### 对接层说明
- CoreML 路径输出与 Vision 路径同为归一化框：`CoordinateMapper / AimEngine / OverlayView` 无需任何修改；
- YOLO11n 约 6MB，A12 及以上芯片单帧约 5–10ms（ANE）；旧机型建议搭配"流畅"运行模式；
- 检测框新增加固：越界框收敛到画面内并剔除无效框、跟踪器 EMA 逐帧平滑（抑制框抖动），两条路径同样生效。

## 视觉设计（灵动岛适配 · 液态玻璃）

所有视觉组件集中在 `Views/Components/LiquidGlass.swift`，功能逻辑零改动。

### 灵动岛适配
- 全画幅预览延伸至灵动岛/状态栏/home indicator 之下；信息层与交互层全部位于安全区内，自动避让系统区域（ZStack 默认安全区布局）；
- `DynamicStatusCapsule`：灵动岛同款黑色胶囊（顶部镜面高光 + 细描边 + plusLighter 混合），位于灵动岛正下方；
- 胶囊四态自适应形变，使用灵动岛同款"果冻感"弹簧曲线（response 0.45 / damping 0.78）：
  - `idle`（瞄准关闭）→ 紧凑圆点；`searching` → 搜索目标；`tracking` → 绿色脉冲点 + 发现人数；`locked` → 展开 214×46 显示锁定状态与 FPS。

### 液态玻璃材质（`LiquidGlassBackground`）
- 结构（克制版）：`ultraThinMaterial` 毛玻璃基底 + 极轻顶亮底沉着色 + 统一发丝描边 + 柔和单层投影；无斜向光带、无渐变镜面描边等装饰性元素；
- `intensity` 参数（0~1）统一驱动玻璃厚度/高光/阴影强度：
  - 设置页导航栏由 `ScrollOffsetKey` 滚动偏移驱动，滚动超过 8pt 后弹性增强（标准 spring 曲线）；
  - 悬浮控制窗 0.95（厚重玻璃）、玻璃卡片 0.85、HUD 芯片 0.7。

### 设置页（重写为玻璃卡片布局）
- 滚动响应液态玻璃导航栏：浮动圆角玻璃条 + 滚动后浮现发丝分隔线；
- 五张 continuous 22pt 圆角玻璃卡片（检测模型/自动瞄准/显示样式/运行状态/恢复默认）；
- 瞄准子选项关闭时以透明度 + 禁用方式收起，卡片高度稳定，过渡平滑；
- 色板选中弹性缩放 + 选中环标记（克制风格，无对勾装饰）；全部控件绑定关系与旧版一致。

### 动效与色彩规范
- 全局动效令牌 `Motion`：标准 spring 0.38/0.82、快速 spring 0.28/0.72、灵动岛 0.45/0.78，无生硬线性过渡；
- 按压反馈统一 `PressableButtonStyle`（弹性缩放 0.95 + 轻降透明度）；
- 色彩全部使用系统语义色（primary/secondary/accent）；标注色板为低饱和 Apple 系统色（白/绿/蓝/橙/红），不含紫色与高饱和装饰色；
- 锁定态视觉语言统一：白色几何（方框/射线/箭头）+ 绿色单一强调（准星、中心点、锁定图标、状态点），全局仅一个彩色强调色；
- 圆角 22/16pt continuous，间距 14pt 网格。

### 兼容性说明
- 以上效果基于 iOS 15 的 Material API 实现（部署目标不变）；iOS 26 原生 `GlassEffectContainer` Liquid Glass API 需要更高部署目标，当前实现为视觉等效方案，未来可平滑替换。

## v2 横屏 ESP 演示模式

- **仅横屏**（LandscapeLeft/Right，默认横屏打开）：采集帧旋转角跟随设备姿态（landscapeLeft=0 / landscapeRight=180，实测校准，iOS 17+ 走 `videoRotationAngle`），Vision 始终按直立帧推理，坐标映射自适应；
- **ESP 标准视觉**：顶部居中射线（origin = 屏幕顶端中点）射向全部已识别目标，锁定目标辉光高亮 + 绿色端点；准星瞄准线保留（实线/虚线/箭头三种样式）；
- **方框样式可调**：粗细（1–8pt）/ 颜色（白/绿/蓝/橙/红）/ 透明度（20%–100%）+ 角框/全框两种形态；
- **灵动岛设置菜单**：顶部胶囊常驻可点，点击以灵动岛同款弹簧曲线展开设置面板（scale 0.12→1 + 淡入，收起反向动画，类似 iOS 应用开合的观感）；胶囊展开态显示"功能设置 · 点击收起"；
- **设置即时生效**：相机采集线程每帧直读设置中心（运行模式/灵敏度），显示开关由 SwiftUI 状态驱动，无传播延迟；
- 旧"悬浮控制窗"已移除（功能并入灵动岛菜单），`FloatingPanel.swift` 删除。
