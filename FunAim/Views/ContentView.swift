import SwiftUI
import UIKit

/// 主界面（横屏 ESP 演示）：
/// 全画幅预览 + 叠加层 + 顶部灵动岛状态胶囊（点击弹性展开设置面板）+ 玻璃 HUD
/// 安全区策略：预览延伸至系统区域之下；信息/交互层位于安全区内。
struct ContentView: View {

    @StateObject private var permission = PermissionManager()
    @ObservedObject private var camera = CameraService.shared
    @StateObject private var aim = AimEngine()
    @ObservedObject private var settings = AppSettings.shared

    @State private var mapped: [MappedPerson] = []
    @State private var viewSize: CGSize = .zero
    @State private var panelExpanded = false
    @State private var previewAngle: CGFloat = 0
    @State private var pinchBase: CGFloat = 1
    @State private var lastLocked = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            switch permission.status {
            case .undetermined: launchView
            case .denied:       deniedView
            case .authorized:   cameraView
            }
        }
        .onAppear(perform: requestPermissionOnLaunch)
        .onChange(of: scenePhase) { phase in
            switch phase {
            case .active:
                permission.refresh()
                if permission.status == .authorized {
                    applyInterfaceOrientation()
                    camera.configureAndStart()
                }
            case .background:
                // 退后台释放摄像头：省电且熄灭系统相机指示灯
                camera.stop()
            default:
                break
            }
        }
        // 物理转动手势即使 UI 横屏锁定也会触发：重新按界面方向同步两端旋转角
        .onReceive(NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification)) { _ in
            applyInterfaceOrientation()
        }
        .onChange(of: settings.flipRotation) { _ in
            applyInterfaceOrientation()
        }
    }

    // MARK: - 相机主界面

    private var cameraView: some View {
        ZStack {
            // 全画幅预览 + 叠加层（延伸至系统区域之下）
            GeometryReader { geo in
                ZStack {
                    // 预览层与数据管线共用同一旋转角（来源：UI 界面方向，系统相机同源方案）
                    CameraPreviewView(session: camera.session, rotationAngle: previewAngle)
                    OverlayView(persons: mapped,
                                reticlePoint: aim.reticlePoint(in: geo.size),
                                targetID: aim.targetID)
                }
                .onAppear {
                    viewSize = geo.size
                    startPipeline()
                }
                .onChange(of: geo.size) { newSize in
                    viewSize = newSize
                }
                // 双指捏合变焦（1x~6x）
                .simultaneousGesture(
                    MagnificationGesture()
                        .onChanged { value in
                            camera.setZoom(pinchBase * value)
                        }
                        .onEnded { _ in
                            pinchBase = camera.currentZoom
                        }
                )
            }
            .ignoresSafeArea()

            // 设置面板（灵动岛展开层）
            if panelExpanded {
                settingsPanel
                    .transition(.scale(scale: 0.12, anchor: .top).combined(with: .opacity))
                    .zIndex(2)
            }

            // 顶部灵动岛状态胶囊（常驻，点击展开/收起设置）
            VStack(spacing: 0) {
                DynamicStatusCapsule(phase: statusPhase,
                                     targetCount: camera.detectionCount,
                                     fps: Int(camera.fps),
                                     isExpanded: panelExpanded,
                                     onTap: {
                                         withAnimation(Motion.island) { panelExpanded.toggle() }
                                     })
                Spacer()
            }
            .padding(.top, 6)
            .zIndex(3)

            // HUD（左上；面板展开时隐藏）
            if settings.showHUD, !panelExpanded {
                hudChip
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(.leading, 14)
                    .padding(.top, 52)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                    .zIndex(1)
            }
            // 闪光灯开关（右上）
            if !panelExpanded {
                Button {
                    camera.toggleTorch()
                } label: {
                    Image(systemName: camera.torchOn ? "flashlight.on.fill" : "flashlight.off.fill")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(camera.torchOn ? .yellow : .primary)
                        .frame(width: 38, height: 38)
                        .liquidGlass(cornerRadius: 19, intensity: 0.7)
                }
                .buttonStyle(PressableButtonStyle())
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                .padding(.trailing, 14)
                .padding(.top, 52)
                .zIndex(1)
            }
        }
        .accentColor(.green)
        .animation(Motion.spring, value: settings.showHUD)
        .animation(Motion.spring, value: camera.torchOn)
        .animation(Motion.island, value: panelExpanded)
        .alert("摄像头不可用",
               isPresented: Binding(
                   get: { camera.lastError != nil },
                   set: { if !$0 { camera.lastError = nil } }
               ),
               actions: {
                   Button("知道了", role: .cancel) {}
               },
               message: { Text(camera.lastError ?? "") })
    }

    // MARK: - 设置面板

    private var settingsPanel: some View {
        SettingsView {
            withAnimation(Motion.island) { panelExpanded = false }
        }
        .frame(width: panelWidth, height: panelHeight)
        .clipShape(RoundedRectangle(cornerRadius: Radii.card, style: .continuous))
        .liquidGlass(cornerRadius: Radii.card, intensity: 0.92)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, 54)
        .padding(.horizontal, 16)
    }

    private var panelWidth: CGFloat {
        min(720, max(420, viewSize.width * 0.62))
    }

    private var panelHeight: CGFloat {
        max(240, viewSize.height - 74)
    }

    // MARK: - 状态推导

    private var statusPhase: DynamicStatusCapsule.Phase {
        if !settings.aimEnabled { return .idle }
        if aim.locked { return .locked }
        return camera.detectionCount > 0 ? .tracking : .searching
    }

    // MARK: - 数据管线装配（相机参数每帧直读设置中心）

    private func startPipeline() {
        applyInterfaceOrientation()
        camera.onDetection = { persons in
            guard viewSize != .zero else { return }
            let mapper = CoordinateMapper(viewSize: viewSize, bufferSize: camera.bufferSize)
            let mappedPersons = persons.map { MappedPerson(person: $0, viewRect: mapper.map($0.box)) }
            mapped = mappedPersons
            updateAim(with: mappedPersons, in: viewSize)
        }
        camera.configureAndStart()
    }

    /// 方向唯一来源：UI 界面方向（系统对"用户怎么握"的最终裁决，与系统相机同源）
    /// 横屏锁定下只有两种取值：interface landscapeRight → 0，interface landscapeLeft → 180
    /// 预览层与数据管线显式设置为同一角度；"画面旋转 180°"开关作为用户手动校正兜底
    private func applyInterfaceOrientation() {
        let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene
        let interface = scene?.interfaceOrientation ?? .landscapeRight
        var angle: CGFloat = (interface == .landscapeLeft) ? 180 : 0
        if settings.flipRotation {
            angle = (angle == 0) ? 180 : 0
        }
        if previewAngle != angle {
            previewAngle = angle
        }
        camera.applyRotationAngle(angle)
    }

    private func updateAim(with persons: [MappedPerson], in size: CGSize) {
        guard settings.aimEnabled else {
            aim.reset()
            if lastLocked {
                lastLocked = false
                UIImpactFeedbackGenerator(style: .light).impactOccurred()
            }
            return
        }
        // 瞄准仅锁定人物（物品只做识别展示）
        let aimable = persons.filter { $0.person.isPerson }
        if let target = aim.selectTarget(from: aimable, viewSize: size, lockStrength: settings.lockStrength) {
            aim.moveReticle(toward: target.center, in: size, tracking: settings.aimTracking)
        } else {
            // 无目标时准星缓慢回归画面中心
            aim.moveReticle(toward: CGPoint(x: size.width / 2, y: size.height / 2),
                            in: size, tracking: 0.03)
        }
        // 锁定/丢失切触觉反馈
        if aim.locked != lastLocked {
            UIImpactFeedbackGenerator(style: aim.locked ? .medium : .light).impactOccurred()
            lastLocked = aim.locked
        }
    }

    // MARK: - 玻璃 HUD

    private var hudChip: some View {
        GlassChip {
            VStack(alignment: .leading, spacing: 3) {
                Text("FunAim · 本地推理")
                    .font(.caption2.bold())
                    .foregroundColor(.secondary)
                Text("FPS \(Int(camera.fps))   推理 \(String(format: "%.0f", camera.inferenceMs))ms   目标 \(camera.detectionCount)")
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.primary)
                if !camera.diagnostics.isEmpty {
                    Text(camera.diagnostics)
                        .font(.caption2.monospacedDigit())
                        .foregroundColor(.yellow.opacity(0.9))
                }
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: - 权限门控视图

    /// 启动页：申请权限期间展示
    private var launchView: some View {
        VStack(spacing: 22) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 58, weight: .light))
                .foregroundColor(.white.opacity(0.9))
                .modifier(PulseEffect(scale: 1.05))
            Text("FunAim 人物锁定演示")
                .font(.title3.bold())
                .foregroundColor(.white)
            ProgressView()
                .tint(.white)
            Text("正在请求摄像头权限…")
                .font(.footnote)
                .foregroundColor(.white.opacity(0.55))
        }
    }

    /// 权限被拒：友好提示 + 引导跳转系统设置
    private var deniedView: some View {
        VStack(spacing: 22) {
            Image(systemName: "video.slash.fill")
                .font(.system(size: 56, weight: .light))
                .foregroundColor(.secondary)
            Text("需要摄像头权限")
                .font(.title2.bold())
            Text("本应用的全部功能都基于摄像头实时画面完成，所有 AI 识别均在手机本地运行，不会录制或上传任何画面。\n\n请前往「设置 → 隐私与安全性 → 摄像头」开启权限后返回。")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 28)

            Button {
                permission.openSystemSettings()
            } label: {
                Label("前往系统设置", systemImage: "gearshape.fill")
                    .font(.body.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 22)
                    .padding(.vertical, 13)
                    .background(Capsule(style: .continuous).fill(Color.green))
            }
            .buttonStyle(PressableButtonStyle())

            Button("我已开启，重新检测") {
                permission.refresh()
            }
            .font(.subheadline.weight(.medium))
            .foregroundColor(.secondary)
            .buttonStyle(PressableButtonStyle())
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.ignoresSafeArea())
    }

    private func requestPermissionOnLaunch() {
        if permission.status == .undetermined {
            permission.request()
        }
    }
}
